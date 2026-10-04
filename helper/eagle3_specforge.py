"""SpecForge EAGLE-3 adapter for the FastGRPO decoder.

The draft architecture and training objective live in SpecForge.  This module
only adapts its one-layer EAGLE-3 module to FastGRPO's existing flat KV-cache
and target-vocabulary decoder contract; it intentionally contains no training
loss implementation.
"""

from __future__ import annotations

import copy
import json
import weakref
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch import nn


SPECFORGE_COMMIT = "3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83"


def rollout_tensor_for_training(tensor: torch.Tensor) -> torch.Tensor:
    """Make inference-only rollout inputs safe to save for draft backward.

    Views, detach(), and same-dtype/device to() do not remove inference status.
    Clone only those inputs, outside inference mode; ordinary inputs keep their
    storage and autograd graph. Values, dtype and device are unchanged.
    """
    if not torch.is_inference(tensor):
        return tensor
    with torch.inference_mode(False):
        return tensor.clone()


def require_specforge():
    try:
        import specforge  # noqa: F401
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from specforge.algorithms.eagle3.model import OnlineEagle3Model
    except Exception as exc:  # pragma: no cover - depends on the runtime image
        raise RuntimeError(
            "SpecForge is required for --draft_backend=eagle3. Install the "
            "commit pinned in requirements-policy-lag.txt; the legacy FastGRPO "
            "draft is not used as a fallback."
        ) from exc
    return AutoDraftModel, AutoDraftModelConfig, OnlineEagle3Model


def resolve_feature_layers(num_hidden_layers: int, configured: Optional[Iterable[int]] = None):
    """Match SpecForge ``resolve_eagle_capture_layers`` exactly."""
    layers = list(configured) if configured is not None else [
        1,
        num_hidden_layers // 2 - 1,
        num_hidden_layers - 4,
    ]
    if len(layers) != 3 or any(not isinstance(i, int) or i < 0 or i >= num_hidden_layers for i in layers):
        raise ValueError(
            f"EAGLE-3 requires exactly three valid target layer ids, got {layers} "
            f"for {num_hidden_layers} layers"
        )
    return layers


def _load_training_state(path: Path):
    if path.is_dir():
        direct = path / "training_state.pt"
        if direct.is_file():
            path = direct
        else:
            latest = sorted(path.glob("*-latest"))
            if len(latest) == 1:
                path = latest[0].resolve() / "training_state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"SpecForge checkpoint state not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    draft_state = state.get("draft_state_dict") if isinstance(state, dict) else None
    if not isinstance(draft_state, dict):
        raise ValueError(f"SpecForge checkpoint has no draft_state_dict: {path}")
    return draft_state, state


class _TargetVocabHead(nn.Module):
    """Expose SpecForge draft-vocab logits as target-token logits."""

    def __init__(self, draft_model):
        super().__init__()
        # Do not register the already-owned draft model a second time under the
        # compatibility head. FastGRPO freezes ``lm_head.parameters()``; a
        # registered reference here would accidentally freeze all EAGLE weights.
        object.__setattr__(self, "_draft_model_ref", weakref.ref(draft_model))

    @property
    def draft_model(self):
        model = self._draft_model_ref()
        if model is None:
            raise RuntimeError("EAGLE-3 draft model was released")
        return model

    def forward(self, hidden_states):
        compact = self.draft_model.compute_logits(hidden_states)
        if compact.shape[-1] == self.draft_model.vocab_size:
            return compact
        target_ids = torch.arange(
            compact.shape[-1], device=compact.device, dtype=torch.long
        ) + self.draft_model.d2t.to(compact.device)
        if torch.unique(target_ids).numel() != target_ids.numel():
            raise RuntimeError("SpecForge d2t mapping is not one-to-one")
        result = compact.new_full(
            (*compact.shape[:-1], self.draft_model.vocab_size),
            torch.finfo(compact.dtype).min,
        )
        return result.scatter(-1, target_ids.expand(*compact.shape[:-1], -1), compact)


class Eagle3FastGRPOAdapter(nn.Module):
    """Minimal EAGLE-3 runtime adapter retaining FastGRPO tree verification.

    SpecForge's training-time cache is depth-oriented. FastGRPO needs a flat
    sequence cache because it prunes verified tree paths. The method below uses
    the *same SpecForge EAGLE-3 weights and layer equations* with a conventional
    one-layer KV cache, allowing FastGRPO's existing pruning code to remain
    unchanged.
    """

    is_eagle3_specforge = True

    def __init__(
        self,
        target_model,
        draft_config: str,
        draft_checkpoint: str = "",
        vocab_mapping: str = "",
        initialization_mode: str = "pretrained",
        feature_layers: Optional[Iterable[int]] = None,
        ttt_length: int = 7,
        lk_loss_type: Optional[str] = None,
        kl_scale: float = 1.0,
        kl_decay: float = 1.0,
    ):
        super().__init__()
        AutoDraftModel, AutoDraftModelConfig, OnlineEagle3Model = require_specforge()
        if initialization_mode not in {"pretrained", "random"}:
            raise ValueError("draft initialization_mode must be pretrained or random")
        if initialization_mode == "pretrained" and not draft_checkpoint:
            raise ValueError("pretrained initialization requires --adapter_path")
        config_path = Path(draft_config)
        if not config_path.is_file():
            raise FileNotFoundError(f"EAGLE-3 draft config not found: {config_path}")
        self.target_model = target_model
        self.config = AutoDraftModelConfig.from_file(str(config_path))
        self.draft_model = AutoDraftModel.from_config(
            self.config,
            attention_backend="sdpa",
            torch_dtype=getattr(target_model, "dtype", torch.bfloat16),
        )
        if draft_checkpoint:
            checkpoint_path = Path(draft_checkpoint)
            is_exported = checkpoint_path.is_dir() and any(
                (checkpoint_path / name).is_file()
                for name in (
                    'model.safetensors', 'model.safetensors.index.json',
                    'pytorch_model.bin', 'pytorch_model.bin.index.json',
                )
            )
            if is_exported:
                self.draft_model = AutoDraftModel.from_pretrained(
                    str(checkpoint_path),
                    config=self.config,
                    attention_backend='sdpa',
                    torch_dtype=getattr(target_model, 'dtype', torch.bfloat16),
                )
            else:
                state, _ = _load_training_state(checkpoint_path)
                missing, unexpected = self.draft_model.load_state_dict(state, strict=False)
                required_missing = [key for key in missing if "embed" not in key.lower()]
                if unexpected or required_missing:
                    raise ValueError(
                        f"incompatible SpecForge draft checkpoint: missing={required_missing}, "
                        f"unexpected={list(unexpected)}"
                    )
        # SpecForge freezes a target-copied embedding. Copy it for random init too.
        target_embedding = target_model.get_input_embeddings().weight
        if self.draft_model.embed_tokens.weight.shape != target_embedding.shape:
            raise ValueError(
                "target/draft embedding shape mismatch: "
                f"{tuple(target_embedding.shape)} vs "
                f"{tuple(self.draft_model.embed_tokens.weight.shape)}"
            )
        with torch.no_grad():
            self.draft_model.embed_tokens.weight.copy_(target_embedding)
        self.draft_model.freeze_embedding()
        if vocab_mapping:
            self.draft_model.load_vocab_mapping(vocab_mapping)
        elif self.draft_model.draft_vocab_size != self.draft_model.vocab_size:
            raise ValueError(
                "a fixed --vocab_mapping is required when draft_vocab_size differs "
                "from target vocab_size"
            )
        else:
            with torch.no_grad():
                self.draft_model.t2d.fill_(True)
                self.draft_model.d2t.zero_()
        mapped_target_ids = torch.arange(self.draft_model.draft_vocab_size) + self.draft_model.d2t.cpu()
        if (
            mapped_target_ids.min().item() < 0
            or mapped_target_ids.max().item() >= self.draft_model.vocab_size
            or torch.unique(mapped_target_ids).numel() != mapped_target_ids.numel()
        ):
            raise ValueError('invalid SpecForge draft-to-target vocabulary mapping')
        selected_target_ids = torch.nonzero(self.draft_model.t2d.cpu(), as_tuple=False).flatten()
        if not torch.equal(selected_target_ids, torch.sort(mapped_target_ids).values):
            raise ValueError('SpecForge t2d and d2t vocabulary mappings disagree')
        n_layers = int(getattr(target_model.config, "num_hidden_layers"))
        self.feature_layers = resolve_feature_layers(n_layers, feature_layers)
        setattr(target_model, "_fastgrpo_eagle3_capture_layers", tuple(self.feature_layers))
        self.specforge_training_model = OnlineEagle3Model(
            self.draft_model,
            length=int(ttt_length),
            attention_backend="sdpa",
            lk_loss_type=lk_loss_type,
            kl_scale=float(kl_scale),
            kl_decay=float(kl_decay),
        )
        self.lm_head = _TargetVocabHead(self.draft_model)
        self.embed_tokens = self.draft_model.embed_tokens
        self.dtype = next(self.draft_model.parameters()).dtype

    @property
    def device(self):
        return next(self.draft_model.parameters()).device

    @property
    def compact_vocab_size(self):
        return int(self.draft_model.draft_vocab_size)

    def compute_compact_logits(self, hidden_states):
        """Return native SpecForge logits without scattering to target vocab."""
        return self.draft_model.compute_logits(hidden_states)

    def compact_to_target_ids(self, compact_ids=None, *, device=None):
        """Map compact EAGLE indices to target tokenizer ids on device."""
        target_ids = torch.arange(
            self.compact_vocab_size,
            device=device or self.device,
            dtype=torch.long,
        ) + self.draft_model.d2t.to(device or self.device)
        return target_ids if compact_ids is None else target_ids[compact_ids]

    def checkpoint_metadata(self):
        return {
            "specforge_commit": SPECFORGE_COMMIT,
            "architecture": type(self.draft_model).__name__,
            "feature_layers": list(self.feature_layers),
            "draft_vocab_size": int(self.draft_model.draft_vocab_size),
            "target_vocab_size": int(self.draft_model.vocab_size),
            "fc_norm": bool(getattr(self.config, 'fc_norm', False)),
        }

    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "specforge_eagle3_fastgrpo_v1",
                "draft_state_dict": self.draft_model.state_dict(),
                "metadata": self.checkpoint_metadata(),
            },
            path,
        )

    def load_model(self, path):
        state, _ = _load_training_state(Path(path))
        self.draft_model.load_state_dict(state, strict=True)

    def _project_feature(self, hidden_states):
        if hidden_states.shape[-1] == self.draft_model.target_hidden_size * 3:
            return self.draft_model.project_hidden_states(hidden_states)
        if hidden_states.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "EAGLE-3 feature width must be 3*target_hidden_size or draft hidden_size; "
                f"got {hidden_states.shape[-1]}"
            )
        return hidden_states

    def forward(
        self,
        hidden_states,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
    ):
        """Run the SpecForge layer equations with FastGRPO-compatible flat KV."""
        draft = self.draft_model
        layer = draft.midlayer
        attn = layer.self_attn
        feature = self._project_feature(hidden_states).to(self.dtype)
        embeds = draft.embed_input_ids(input_ids).to(self.dtype)
        residual = feature
        mixed = torch.cat(
            (layer.input_layernorm(embeds), layer.hidden_norm(feature)), dim=-1
        )
        bsz, q_len, _ = mixed.shape
        query = attn.q_proj(mixed).view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
        key = attn.k_proj(mixed).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        value = attn.v_proj(mixed).view(bsz, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        past_len = 0 if not past_key_values else past_key_values[0][0].shape[-2]
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + q_len, device=mixed.device, dtype=torch.long,
            ).unsqueeze(0)
        else:
            # RoPE indexes its cos/sin tables with these IDs. Legacy Model
            # already normalizes supplied IDs to long; preserve that contract
            # here too without casting positions to the BF16 model dtype.
            position_ids = position_ids.to(device=mixed.device, dtype=torch.long)
        cos, sin = attn.rotary_emb(value, seq_len=past_len + q_len)
        # Import the exact rotary/repeat helpers used by the pinned SpecForge model.
        from specforge.modeling.draft.llama3_eagle import apply_rotary_pos_emb, repeat_kv

        query, key = apply_rotary_pos_emb(query, key, cos.to(query.device), sin.to(query.device), position_ids)
        if past_key_values:
            key = torch.cat((past_key_values[0][0], key), dim=-2)
            value = torch.cat((past_key_values[0][1], value), dim=-2)
        present = [[key, value]] if use_cache else []
        key_rep = repeat_kv(key, attn.num_key_value_groups)
        value_rep = repeat_kv(value, attn.num_key_value_groups)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key_rep,
            value_rep,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attention_mask is None and past_len == 0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        feature = residual + attn.o_proj(attended)
        residual = feature
        feature = residual + layer.mlp(layer.post_attention_layernorm(feature))
        # EAGLE-3 recursively feeds its predicted feature into the next draft step.
        return {
            "hidden_states": feature,
            "next_feature_states": feature,
            "past_key_values": present,
        }


def capture_eagle3_features(outputs, layer_ids):
    """Return (three-layer concatenation, final hidden) from an HF forward."""
    if outputs.hidden_states is None:
        raise RuntimeError("target forward did not return hidden_states")
    # HF hidden_states[0] is the embedding output; SpecForge layer ids address
    # decoder layers, hence +1.
    aux = torch.cat([outputs.hidden_states[int(i) + 1] for i in layer_ids], dim=-1)
    return aux, outputs.last_hidden_state


def clone_state_dict(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def clone_optimizer_state(optimizer):
    return copy.deepcopy(optimizer.state_dict())

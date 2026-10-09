from typing import Optional

import torch
import torch.nn as nn

from specforge.modeling.target.checkpoint import (
    load_tensors_by_keys,
    resolve_checkpoint_dir,
)
from specforge.modeling.target.target_utils import (
    load_target_config,
    target_hidden_size,
    target_text_config,
    target_vocab_size,
)
from specforge.utils import get_local_device, padding


class TargetHead(nn.Module):
    def __init__(
        self,
        model_path,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.config = load_target_config(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        self.hidden_size = target_hidden_size(self.config)
        self.vocab_size = target_vocab_size(self.config)

        self.fc = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        lm_head_key: str = "lm_head.weight",
        embedding_key: str = "model.embed_tokens.weight",
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> "TargetHead":
        target_head = cls(
            model_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        target_head.load_weights(
            model_path=model_path,
            lm_head_key=lm_head_key,
            embedding_key=embedding_key,
            cache_dir=cache_dir,
        )
        target_head.freeze_weights()
        target_head = target_head.eval().to(
            device=get_local_device(), dtype=torch.bfloat16
        )
        return target_head

    @torch.no_grad()
    def load_weights(
        self,
        model_path,
        lm_head_key: str = "lm_head.weight",
        embedding_key: str = "model.embed_tokens.weight",
        cache_dir: Optional[str] = None,
    ):
        self.model_path = resolve_checkpoint_dir(
            model_path,
            cache_dir=cache_dir,
            allow_patterns=["*.json", "*.safetensors", "*.bin"],
        )
        text_config = target_text_config(self.config)
        tie_weights = bool(
            getattr(
                text_config,
                "tie_word_embeddings",
                getattr(self.config, "tie_word_embeddings", False),
            )
        )
        candidate_keys = [lm_head_key]
        if tie_weights and embedding_key != lm_head_key:
            candidate_keys.append(embedding_key)
        tensors = load_tensors_by_keys(self.model_path, candidate_keys)

        resolved_key = lm_head_key
        if lm_head_key not in tensors:
            if tie_weights and embedding_key in tensors:
                resolved_key = embedding_key
                print(
                    "Tied target embeddings detected: loading TargetHead from "
                    f"{embedding_key!r} because {lm_head_key!r} is not stored "
                    "separately."
                )
            else:
                raise KeyError(
                    f"Target head tensor {lm_head_key!r} is missing from "
                    f"{self.model_path!r}; tie_word_embeddings={tie_weights}, "
                    f"embedding fallback {embedding_key!r} present="
                    f"{embedding_key in tensors}"
                )

        lm_head = tensors[resolved_key]
        if tuple(lm_head.shape) != tuple(self.fc.weight.shape):
            raise RuntimeError(
                f"Target head shape mismatch for {resolved_key!r}: expected "
                f"{tuple(self.fc.weight.shape)}, got {tuple(lm_head.shape)}"
            )
        self.fc.weight.copy_(lm_head)

    def freeze_weights(self):
        for param in self.fc.parameters():
            param.requires_grad = False

    def forward(self, hidden_states):
        return self.fc(hidden_states)

    def preprocess(self, input_ids, target, loss_mask):
        # Shift loss_mask with input_ids and final target states; auxiliary hidden
        # states remain unshifted because the draft model consumes them at row p.
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)
        loss_mask = padding(loss_mask, left=False)[..., None]
        return input_ids, target, loss_mask

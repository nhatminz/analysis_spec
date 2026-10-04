"""Trajectory-local, optimizer-free LK Reflex state for EAGLE-3 rollout.

The class in this file is deliberately a plain Python object rather than an
``nn.Module``.  Its tensors are inference-time state: they are never model
parameters, never enter an optimizer, and never survive a rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Optional
import importlib
import importlib.util
import time

import torch
import torch.nn.functional as F


_PROJECTION_CACHE: dict[tuple[str, int, int, int], torch.Tensor] = {}


def resolve_reflex_backend(requested, device, feature_dim):
    """Resolve once per rollout; do not import Triton on the CPU/torch path."""
    if requested not in {"auto", "torch", "triton"}:
        raise ValueError("Reflex backend must be auto, torch or triton")
    if requested == "torch":
        return "torch"
    supported = torch.device(device).type == "cuda" and int(feature_dim) <= 64
    available = supported and importlib.util.find_spec("triton") is not None
    if requested == "triton" and not available:
        raise RuntimeError("Reflex triton backend requires CUDA, Triton and feature_dim <= 64; choose torch")
    return "triton" if requested != "torch" and available else "torch"


def lk_alpha_and_logit_gradient_without_loss(
    q: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the alpha and analytic gradient required by the update."""
    if q.shape != p.shape:
        raise ValueError(f"p/q shape mismatch: p={tuple(p.shape)}, q={tuple(q.shape)}")
    q32 = q.float()
    p32 = p.float()
    alpha = torch.minimum(p32, q32).sum(dim=-1)
    mask = (q32 < p32).to(q32.dtype)
    selected_mass = (mask * q32).sum(dim=-1, keepdim=True)
    gradient = q32 * (selected_mass - mask) / (alpha.unsqueeze(-1) + float(eps))
    return alpha, gradient


def lk_diagnostic_loss(alpha: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute the optional LK scalar used only when diagnostics are enabled."""
    return -(alpha + float(eps)).log()


def lk_alpha_and_logit_gradient(
    q: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return LK alpha, loss and analytic ``d loss / d logits(q)``.

    ``q`` must be a softmax distribution and ``p`` the teacher probability on
    the same vocabulary.  All reductions are batched over the last dimension.
    At the non-differentiable equality point we consistently choose ``m=0``.
    """
    alpha, gradient = lk_alpha_and_logit_gradient_without_loss(q, p, eps)
    loss = lk_diagnostic_loss(alpha, eps)
    return alpha, loss, gradient


@dataclass(frozen=True)
class ReflexStats:
    alpha_sum: Optional[float]
    loss_sum: Optional[float]
    updates: int
    profile_time_ms: float = 0.0
    profile_sections_ms: Optional[dict] = None
    update_gpu_ms: Optional[float] = None


class FastLKReflex:
    """Low-rank-in-context fast state ``z = z0 + A @ psi``.

    A is stored in FP32 as ``[active_trajectory, compact_vocab, feature_dim]``.
    R and A have ``requires_grad=False`` and this class exposes no parameters.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        learning_rate: float = 0.05,
        weight_decay: float = 0.0,
        seed: int = 42,
        eps: float = 1.0e-8,
        profile: bool = False,
        diagnostics: bool = False,
        backend: str = "auto",
        feedback_scope: str = "root",
        proposal_strategy: str = "fused",
        correction_strategy: str = "serial",
        feedback_strategy: str = "serial",
        feature_strategy: str = "auto",
        time_updates: bool = False,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if learning_rate < 0.0 or weight_decay < 0.0:
            raise ValueError("learning_rate and weight_decay must be non-negative")
        self.feature_dim = int(feature_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.eps = float(eps)
        self.profile = bool(profile)
        self.time_updates = bool(time_updates)
        self.diagnostics = bool(diagnostics)
        if backend not in {"auto", "torch", "triton"}:
            raise ValueError("Reflex backend must be auto, torch or triton")
        self.requested_backend = backend
        if feedback_scope not in {"root", "visited_path"}:
            raise ValueError("feedback_scope must be root or visited_path")
        self.feedback_scope = feedback_scope
        if proposal_strategy not in {"fused", "sort", "hybrid", "torch"}:
            raise ValueError("proposal_strategy must be fused, sort, hybrid or torch")
        if correction_strategy not in {"serial", "parallel", "tiled"}:
            raise ValueError("correction_strategy must be serial, parallel or tiled")
        if feedback_strategy not in {"serial", "parallel"}:
            raise ValueError("feedback_strategy must be serial or parallel")
        if feature_strategy not in {"auto", "triton", "torch"}:
            raise ValueError("feature_strategy must be auto, triton or torch")
        self.proposal_strategy = proposal_strategy
        self.correction_strategy = correction_strategy
        self.feedback_strategy = feedback_strategy
        self.feature_strategy = feature_strategy
        self._round_contexts = 0
        self._feedback_raw = self._feedback_psi = self._feedback_norm = None
        self._proposal_workspace = self._feedback_workspace = self.path_workspace = self._root_indices = None
        self.padded_path_workspace = None
        self._corrected_workspace = None
        self.backend = "torch"
        self._kernels = None
        self.projection: Optional[torch.Tensor] = None
        self.state: Optional[torch.Tensor] = None
        self._state_workspace: Optional[torch.Tensor] = None
        self._root_q: Optional[torch.Tensor] = None
        self._root_psi: Optional[torch.Tensor] = None
        self._alpha_sum: Optional[torch.Tensor] = None
        self._loss_sum: Optional[torch.Tensor] = None
        self._updates = 0
        self._profile_time_s = 0.0
        self._profile_events = []

    def _gpu_profile_start(self):
        if not (self.profile or self.time_updates) or self.state is None or not self.state.is_cuda:
            return None
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _gpu_profile_end(self, label, started):
        if started is not None:
            ended = torch.cuda.Event(enable_timing=True)
            ended.record()
            self._profile_events.append((label, started, ended))

    def _profile_sections(self):
        totals = {}
        for label, started, ended in self._profile_events:
            ended.synchronize()  # opt-in profiling, only after the rollout
            totals[label] = totals.get(label, 0.0) + started.elapsed_time(ended)
        return totals

    def _profile_start(self):
        if not self.profile:
            return None
        return time.perf_counter()

    def _profile_end(self, started) -> None:
        if started is None:
            return
        self._profile_time_s += time.perf_counter() - started

    @property
    def active_trajectories(self) -> int:
        return 0 if self.state is None else int(self.state.shape[0])

    def start(
        self,
        num_trajectories: int,
        compact_vocab_size: int,
        hidden_size: int,
        device: torch.device | str,
        *, max_contexts=1, max_path_length=1, max_proposal_contexts=8, max_topk=8,
    ) -> None:
        """Create one zero fast state per response, once per rollout."""
        if num_trajectories <= 0 or compact_vocab_size <= 0 or hidden_size <= 0:
            raise ValueError("trajectory, vocabulary, and hidden sizes must be positive")
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.backend = resolve_reflex_backend(self.requested_backend, device, self.feature_dim)
        self._kernels = (
            importlib.import_module("helper.fast_lk_reflex_kernels")
            if self.backend == "triton" else None
        )
        # Initialization is outside the per-round hot path. A device-local
        # generator makes R reproducible without copying it from CPU.
        cache_key = (str(device), hidden_size, self.feature_dim, self.seed)
        projection = _PROJECTION_CACHE.get(cache_key)
        if projection is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.seed)
            projection = torch.randn(
                (hidden_size, self.feature_dim),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            projection.mul_(hidden_size ** -0.5)
            projection.requires_grad_(False)
            _PROJECTION_CACHE[cache_key] = projection
        self.projection = projection
        self.state = torch.zeros(
            (num_trajectories, compact_vocab_size, self.feature_dim),
            device=device,
            dtype=torch.float32,
            requires_grad=False,
        )
        # Double-buffer only the small fast adapter, not the model or KV cache.
        # Finished-row compaction can reuse storage without a fresh A allocation.
        self._state_workspace = torch.empty_like(self.state)
        self._root_q = None
        self._root_psi = None
        self._alpha_sum = (
            torch.zeros((), device=device, dtype=torch.float32)
            if self.diagnostics else None
        )
        self._loss_sum = (
            torch.zeros((), device=device, dtype=torch.float32)
            if self.diagnostics else None
        )
        self._updates = 0
        self._profile_time_s = 0.0
        self._profile_events = []
        self._batch_capacity = num_trajectories
        self._context_capacity = int(max_contexts) if self.feedback_scope == "visited_path" else 1
        self._path_capacity = int(max_path_length)
        if min(self._context_capacity, self._path_capacity, int(max_proposal_contexts), int(max_topk)) <= 0:
            raise ValueError("Reflex workspace capacities must be positive")
        self._round_contexts = 0
        self._feedback_raw = self._feedback_psi = self._feedback_norm = None
        self.path_workspace = [torch.empty((num_trajectories, self._path_capacity), device=device, dtype=torch.long)
                               for _ in range(3)] + [torch.empty(num_trajectories, device=device, dtype=torch.long)]
        self.padded_path_workspace = [torch.empty((num_trajectories, self._path_capacity),
                                                   device=device, dtype=torch.bool if index == 2 else torch.long)
                                      for index in range(3)]
        self.padded_path_workspace.append(torch.empty((num_trajectories, 1), device=device, dtype=torch.long))
        self._root_indices = torch.zeros((num_trajectories, 1), device=device, dtype=torch.long)
        self._proposal_workspace = self._feedback_workspace = None
        self._corrected_workspace = None
        if self._kernels is not None:
            tiles = (compact_vocab_size + 255) // 256
            base = num_trajectories * int(max_proposal_contexts) * tiles
            self._proposal_workspace = [torch.empty(base * (int(max_topk) if i >= 2 else 1),
                device=device, dtype=torch.long if i == 3 else torch.float32) for i in range(4)]
            feedback_size = num_trajectories * self._path_capacity * tiles
            self._feedback_workspace = [torch.empty(feedback_size * factor, device=device, dtype=torch.float32)
                                        for factor in (1, 2)]
            self._feedback_workspace.append(torch.empty(num_trajectories * self._path_capacity,
                                                        device=device, dtype=torch.float32))
            if self.proposal_strategy == "hybrid":
                self._corrected_workspace = torch.empty(
                    num_trajectories * int(max_proposal_contexts) * compact_vocab_size,
                    device=device, dtype=torch.float32)

    def _cache_contexts(self, raw, psi, norm):
        """Native logits only (usually BF16), psi and max/sum normalization.

        Pools survive rounds and finished-row compaction. Every next draft round
        overwrites its active rows; no old-cache compaction/copy is necessary.
        """
        batch, contexts, vocab = raw.shape
        start, end = self._round_contexts, self._round_contexts + contexts
        if end > self._context_capacity:
            raise ValueError("feedback context capacity exceeded; pass max_contexts to start()")
        if self._feedback_raw is None:
            shape = (self._batch_capacity, self._context_capacity)
            self._feedback_raw = torch.empty((*shape, vocab), device=raw.device, dtype=raw.dtype)
            self._feedback_psi = torch.empty((*shape, self.feature_dim), device=raw.device, dtype=torch.float32)
            self._feedback_norm = torch.empty((*shape, 2), device=raw.device, dtype=torch.float32)
        elif self._feedback_raw.dtype != raw.dtype:
            raise ValueError("proposal logits dtype must remain fixed within a rollout")
        self._feedback_raw[:batch, start:end].copy_(raw)
        self._feedback_psi[:batch, start:end].copy_(psi)
        self._feedback_norm[:batch, start:end].copy_(norm)
        self._round_contexts = end

    @torch.no_grad()
    def propose(self, compact_logits, native_hidden, k, mapping, *, root=False):
        """Top-k without dense branch probabilities on the Triton backend."""
        started = self._profile_start()
        if self.state is None or compact_logits.shape[0] != self.active_trajectories:
            raise ValueError("Reflex state is not aligned with the active batch")
        k = int(k)
        if not 1 <= k <= compact_logits.shape[-1]:
            raise ValueError("invalid proposal top-k")
        if root:
            if compact_logits.shape[1] != 1:
                raise ValueError("root proposal must contain one context")
            self._round_contexts = 0
            self._root_q = self._root_psi = None
        needs_fp32_guard = (self._kernels is None or self.proposal_strategy in {"hybrid", "torch"}
                            or self.feature_strategy == "torch")
        guard = (torch.autocast(device_type=compact_logits.device.type, enabled=False)
                 if needs_fp32_guard and torch.is_autocast_enabled(compact_logits.device.type)
                 else nullcontext())
        with guard:
            feature_event = self._gpu_profile_start() if self.profile else None
            psi = self._feature(native_hidden)
            if self.profile:
                self._gpu_profile_end("feature_projection_ms", feature_event)
            proposal_event = self._gpu_profile_start() if self.profile else None
            if self._kernels is not None and self.proposal_strategy in {"fused", "sort"}:
                values, ids, norm = self._kernels.propose(compact_logits, psi, self.state, k,
                                                        self._proposal_workspace,
                                                        strategy=self.proposal_strategy)
            else:
                if self._kernels is not None and self.proposal_strategy == "hybrid":
                    corrected_size = compact_logits.numel()
                    z = self._kernels.correct_logits(compact_logits, psi, self.state,
                                                      strategy=self.correction_strategy,
                                                      out=self._corrected_workspace[:corrected_size].view(
                                                          compact_logits.shape))
                else:
                    z = torch.baddbmm(compact_logits.float(), psi, self.state.transpose(1, 2))
                q = z.softmax(dim=-1)
                values, ids = torch.topk(q, k=k, dim=-1)
                norm = None
                if self.feedback_scope == "visited_path" or (root and self._kernels is not None):
                    maximum = z.amax(-1)
                    norm = torch.stack((maximum, (z - maximum.unsqueeze(-1)).exp().sum(-1)), -1)
                if root and self.feedback_scope == "root":
                    # Keep the exact existing Torch/root ablation, no reconstruction.
                    self._root_q, self._root_psi = q.squeeze(1), psi.squeeze(1)
            if self.profile:
                self._gpu_profile_end("proposal_ms", proposal_event)
        if self.feedback_scope == "visited_path" or (root and self._kernels is not None):
            self._cache_contexts(compact_logits, psi, norm)
        self._profile_end(started)
        return values, ids, mapping[ids]

    @torch.no_grad()
    def update_visited(self, target, mapping, path, *, greedy=False, validate=False):
        """One state write per round; mean over eligible visited proposal heads."""
        started = self._profile_start()
        update_event = self._gpu_profile_start()
        if self._feedback_raw is None or self._round_contexts == 0:
            raise RuntimeError("proposal feedback was not cached")
        batch = self.active_trajectories
        contexts, indices = path.feedback_contexts, path.packed_indices
        if contexts.shape != indices.shape or indices.ndim != 2 or indices.shape[0] != batch:
            raise ValueError("feedback path is not aligned with the active batch")
        if indices.shape[1] > self._path_capacity:
            raise ValueError("feedback path capacity exceeded")
        # Production paths come from our bounded tensor verifier. Do NOT launch
        # separate validation/reduction kernels per update in the default path.
        # Explicit debug validation stays asynchronous (no GPU scalar on host).
        valid = (contexts >= 0) & (indices >= 0) if self._kernels is None or self.diagnostics or validate else None
        if validate:
            torch._assert_async(((contexts[:, 0] == 0) & (indices[:, 0] == 0)).all(),
                                "feedback path must start at the root")
            torch._assert_async(((contexts < self._round_contexts) | ~valid).all(), "invalid feedback context")
            torch._assert_async(((indices < target.shape[1]) | ~valid).all(), "invalid target row")
        raw, psi, norm = self._feedback_raw[:batch], self._feedback_psi[:batch], self._feedback_norm[:batch]
        guard = (torch.autocast(device_type=self.state.device.type, enabled=False)
                 if self._kernels is None and torch.is_autocast_enabled(self.state.device.type)
                 else nullcontext())
        with guard:
            if self._kernels is not None:
                alpha = self._kernels.update_path(self.state, raw, psi, norm, target, mapping,
                    indices, contexts, greedy=greedy, eps=self.eps, learning_rate=self.learning_rate,
                    decay=1.0 - self.learning_rate * self.weight_decay, workspace=self._feedback_workspace,
                    strategy=self.feedback_strategy)
            else:
                rows = torch.arange(batch, device=self.state.device)[:, None]
                safe = contexts.clamp_min(0)
                features = psi[rows, safe]
                z = torch.baddbmm(raw[rows, safe].float(), features, self.state.transpose(1, 2))
                norms = norm[rows, safe]
                q = (z - norms[..., :1]).exp() / norms[..., 1:]
                teacher = target[rows, indices.clamp_min(0)]
                p = teacher.unsqueeze(-1).eq(mapping).float() if greedy else teacher.float().index_select(-1, mapping)
                p = p / (p.sum(-1, keepdim=True) + self.eps)
                alpha, gradient = lk_alpha_and_logit_gradient_without_loss(q, p, self.eps)
                counts = valid.sum(1).clamp_min(1).float()
                gradient = gradient * valid.unsqueeze(-1) / counts[:, None, None]
                self.state.baddbmm_(gradient.transpose(1, 2), features,
                    beta=1.0 - self.learning_rate * self.weight_decay, alpha=-self.learning_rate)
                alpha = alpha * valid
        # Diagnostics retain their per-trajectory denominator; average contexts.
        loss = None
        if self.diagnostics:
            counts = valid.sum(1).clamp_min(1)
            mean_alpha = alpha.sum(1) / counts
            loss = (lk_diagnostic_loss(alpha, self.eps) * valid).sum(1) / counts
            self._record_update(mean_alpha, loss)
        else:
            self._record_update(alpha[:, 0], None)  # shape only; no diagnostic reductions
        self._round_contexts = 0
        self._profile_end(started)
        if self.profile or self.time_updates:
            self._gpu_profile_end("feedback_update_ms", update_event)
        return alpha, loss

    def _update_cached_root(self, target, mapping, *, greedy):
        from helper.tree_verification import VerifiedPath
        zeros = self._root_indices[:self.active_trajectories]
        path = VerifiedPath(zeros, zeros, zeros, zeros[:, 0])
        alpha, loss = self.update_visited(target.unsqueeze(1), mapping, path, greedy=greedy)
        return alpha[:, 0], loss

    def _feature(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.projection is None:
            raise RuntimeError("FastLKReflex.start() must be called before correction")
        if self._kernels is not None:
            return self._kernels.feature(hidden_states, self.projection, strategy=self.feature_strategy)
        return F.normalize(hidden_states.float().matmul(self.projection), dim=-1, eps=1.0e-6)

    @torch.no_grad()
    def correct(
        self,
        compact_logits: torch.Tensor,
        native_hidden: torch.Tensor,
        *,
        cache_root: bool = False,
    ) -> torch.Tensor:
        """Apply A@psi and return probabilities; optionally cache root q/psi."""
        started = self._profile_start()
        if self.state is None:
            raise RuntimeError("FastLKReflex.start() must be called before correction")
        if compact_logits.shape[0] != self.state.shape[0]:
            raise ValueError("Reflex state is not aligned with the active batch")
        # A/R/psi and the analytic update are FP32, irrespective of model AMP.
        # Otherwise autocast copies the entire A to BF16 at every tree depth
        # and cached BF16 psi cannot be used by the FP32 in-place update.
        # Pure Triton already performs explicit FP32 arithmetic; avoid entering
        # an extra AMP context on its hot path. Torch needs the guard only when
        # the caller actually enabled model autocast.
        guard = (torch.autocast(device_type=compact_logits.device.type, enabled=False)
                 if (self._kernels is None or self.feature_strategy == "torch")
                 and torch.is_autocast_enabled(compact_logits.device.type)
                 else nullcontext())
        with guard:
            feature_event = self._gpu_profile_start() if self.profile else None
            psi = self._feature(native_hidden)
            if self.profile:
                self._gpu_profile_end("feature_projection_ms", feature_event)
            correction_event = self._gpu_profile_start() if self.profile else None
            if self._kernels is not None:
                probabilities = self._kernels.correct(compact_logits, psi, self.state)
            else:
                corrected = torch.baddbmm(
                    compact_logits.float(), psi, self.state.transpose(1, 2)
                )
                probabilities = corrected.softmax(dim=-1)
            if self.profile:
                self._gpu_profile_end("correction_ms", correction_event)
        if cache_root:
            if probabilities.shape[1] != 1:
                raise ValueError("root correction expects exactly one proposal context")
            self._root_q = probabilities.squeeze(1)
            self._root_psi = psi.squeeze(1)
        self._profile_end(started)
        return probabilities

    @torch.no_grad()
    def update_from_target_probs(
        self,
        target_root_probs: torch.Tensor,
        compact_to_target: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Perform one root-only analytic LK update using verification logits."""
        if self._root_q is None and self._feedback_raw is not None:
            return self._update_cached_root(target_root_probs, compact_to_target, greedy=False)
        if self.state is None or self._root_q is None or self._root_psi is None:
            raise RuntimeError("root q/psi were not cached before the Reflex update")
        mapping = compact_to_target.to(device=target_root_probs.device, dtype=torch.long)
        if self._kernels is not None:
            return self._update_fused(target_root_probs, mapping, greedy=False)
        # Conditional compact-vocabulary LK: condition the exact target sampling
        # distribution on tokens controllable by the EAGLE compact head.
        p = target_root_probs.float().index_select(-1, mapping)
        p = p / (p.sum(dim=-1, keepdim=True) + self.eps)
        return self._update_from_compact_probs(p)

    @torch.no_grad()
    def update_from_target_tokens(
        self,
        target_root_tokens: torch.Tensor,
        compact_to_target: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply greedy one-hot supervision without a full-vocabulary tensor."""
        if self._root_q is None and self._feedback_raw is not None:
            return self._update_cached_root(target_root_tokens, compact_to_target, greedy=True)
        if self.state is None or self._root_q is None or self._root_psi is None:
            raise RuntimeError("root q/psi were not cached before the Reflex update")
        mapping = compact_to_target.to(
            device=target_root_tokens.device, dtype=torch.long
        )
        if self._kernels is not None:
            return self._update_fused(target_root_tokens.to(torch.long), mapping, greedy=True)
        p = target_root_tokens.to(torch.long).unsqueeze(-1).eq(mapping.unsqueeze(0))
        p = p.to(torch.float32)
        p = p / (p.sum(dim=-1, keepdim=True) + self.eps)
        return self._update_from_compact_probs(p)

    def _update_from_compact_probs(
        self,
        compact_target_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Shared in-place update after compact target supervision is formed."""
        started = self._profile_start()
        update_event = self._gpu_profile_start()
        p = compact_target_probs
        alpha, gradient = lk_alpha_and_logit_gradient_without_loss(
            self._root_q, p, self.eps
        )
        loss = lk_diagnostic_loss(alpha, self.eps) if self.diagnostics else None
        decay = 1.0 - self.learning_rate * self.weight_decay
        # baddbmm_ applies decay and the batched rank-one update without
        # materializing a [batch, vocabulary, feature] outer-product temporary.
        self.state.baddbmm_(
            gradient.unsqueeze(-1),
            self._root_psi.float().unsqueeze(1),
            beta=decay,
            alpha=-self.learning_rate,
        )
        self._record_update(alpha, loss)
        self._profile_end(started)
        if self.profile or self.time_updates:
            self._gpu_profile_end("feedback_update_ms", update_event)
        return alpha, loss

    def _update_fused(self, target, mapping, *, greedy):
        started = self._profile_start()
        update_event = self._gpu_profile_start()
        alpha = self._kernels.update(
            self.state, self._root_q, self._root_psi, target, mapping,
            greedy=greedy, eps=self.eps, learning_rate=self.learning_rate,
            decay=1.0 - self.learning_rate * self.weight_decay,
        )
        loss = lk_diagnostic_loss(alpha, self.eps) if self.diagnostics else None
        self._record_update(alpha, loss)
        self._profile_end(started)
        if self.profile or self.time_updates:
            self._gpu_profile_end("feedback_update_ms", update_event)
        return alpha, loss

    def _record_update(self, alpha, loss):
        if self.diagnostics:
            self._alpha_sum.add_(alpha.sum())
            self._loss_sum.add_(loss.sum())
        self._updates += int(alpha.shape[0])
        self._root_q = None
        self._root_psi = None

    @torch.no_grad()
    def remove_finished(self, finished_indices) -> None:
        """Compact all completed responses once while preserving batch order."""
        if self.state is None:
            raise RuntimeError("FastLKReflex has not been started")
        if not finished_indices:
            return
        finished_set = {int(index) for index in finished_indices}
        if any(index < 0 or index >= self.state.shape[0] for index in finished_set):
            raise IndexError(finished_indices)
        keep_indices = [
            index for index in range(self.state.shape[0]) if index not in finished_set
        ]
        keep = torch.tensor(keep_indices, device=self.state.device, dtype=torch.long)
        previous = self.state
        compacted = self._state_workspace[:len(keep_indices)]
        torch.index_select(previous, 0, keep, out=compacted)
        self.state = compacted
        self._state_workspace = previous
        # A root cache belongs to the just-verified batch and has already been
        # consumed. Clearing defensively prevents accidental cross-round reuse.
        self._root_q = None
        self._root_psi = None

    def finish(self) -> ReflexStats:
        """Materialize aggregate scalars once, at rollout completion."""
        sections = self._profile_sections() if (self.profile or self.time_updates) else None
        update_ms = sections.get("feedback_update_ms", 0.0) if sections is not None else None
        if not self.diagnostics:
            return ReflexStats(
                alpha_sum=None, loss_sum=None, updates=int(self._updates),
                profile_time_ms=self._profile_time_s * 1000.0,
                profile_sections_ms=sections, update_gpu_ms=update_ms,
            )
        if self._updates == 0 or self._alpha_sum is None or self._loss_sum is None:
            return ReflexStats(
                alpha_sum=0.0, loss_sum=0.0, updates=0,
                profile_time_ms=self._profile_time_s * 1000.0,
                profile_sections_ms=sections, update_gpu_ms=update_ms,
            )
        return ReflexStats(
            alpha_sum=float(self._alpha_sum.item()),
            loss_sum=float(self._loss_sum.item()),
            updates=int(self._updates),
            profile_time_ms=self._profile_time_s * 1000.0,
            profile_sections_ms=sections, update_gpu_ms=update_ms,
        )

    def clear(self) -> None:
        self._feedback_raw = self._feedback_psi = self._feedback_norm = None
        self._proposal_workspace = self._feedback_workspace = self.path_workspace = self._root_indices = None
        self.padded_path_workspace = None
        self._corrected_workspace = None
        self._round_contexts = 0
        self.projection = None
        self.state = None
        self._state_workspace = None
        self._kernels = None
        self._root_q = None
        self._root_psi = None
        self._alpha_sum = None
        self._loss_sum = None
        self._updates = 0
        self._profile_time_s = 0.0
        self._profile_events = []


def reflex_or_baseline_probabilities(
    raw_logits: torch.Tensor,
    native_hidden: Optional[torch.Tensor] = None,
    reflex: Optional[FastLKReflex] = None,
    *,
    cache_root: bool = False,
) -> torch.Tensor:
    """Use the same FP32 compact-proposal path in OFF and ACTIVE modes."""
    if reflex is None:
        return raw_logits.float().softmax(dim=-1)
    if native_hidden is None:
        raise ValueError("native_hidden is required when Reflex is active")
    return reflex.correct(raw_logits, native_hidden, cache_root=cache_root)


def topk_compact_candidates(probabilities, compact_to_target, k):
    """Shared OFF/ACTIVE compact top-k and fixed d2t mapping path."""
    values, compact_ids = torch.topk(probabilities, k=int(k), dim=-1)
    mapping = compact_to_target.to(probabilities.device, torch.long)
    return values, compact_ids, mapping[compact_ids]

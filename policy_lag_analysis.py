#!/usr/bin/env python3
"""Supervision-lag experiment utilities and result exporter.

The production run is executed by ``grpo_speculative.py`` with its EAGLE-3
backend; this module owns invariant checks, exact metrics, resumable boundary
journals and the dependency/CLI smoke test used by the launcher.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SPECFORGE_COMMIT = "3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83"
FASTGRPO_COMMIT = "38e252493149072d2c5905f0a47de1d935d7170a"
PROTOCOL_VERSION = "fastgrpo_policy_lag_protocol_v5"
BRANCHES = ("stale", "fresh", "reflex")
STEP_METRIC_COLUMNS = (
    "policy_step", "stale_aal", "fresh_aal", "reflex_aal",
    "fresh_minus_stale_aal", "reflex_minus_stale_aal",
    "stale_accepted_length_sum", "stale_verification_rounds",
    "fresh_accepted_length_sum", "fresh_verification_rounds",
    "reflex_accepted_length_sum", "reflex_verification_rounds", "teacher_shift_tv",
    "fresh_minus_stale_ci_low", "fresh_minus_stale_ci_high",
    "reflex_minus_stale_ci_low", "reflex_minus_stale_ci_high",
    "requested_training_token_budget", "actual_training_token_budget",
    "draft_optimizer_steps", "effective_draft_lr",
    "stale_online_draft_update_gpu_ms", "fresh_online_draft_update_gpu_ms",
    "reflex_update_gpu_ms", "reflex_update_count", "reflex_update_gpu_ms_per_update",
    "reflex_update_exposed_ms", "analysis_io_wall_ms",
)


class AnalysisIO:
    """CPU wall time of serialization/writes only, never GPU computation.

    Materialize CPU tensor snapshots BEFORE entering measure(). The tiny final
    io_timing.json write is excluded to avoid self-referential timing.
    """
    def __init__(self):
        self.wall_ms = 0.0

    @contextmanager
    def measure(self):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.wall_ms += (time.perf_counter() - started) * 1000.0


def cleanup_branch_checkpoints(output_dir, durable_target_step, *, keep=False):
    """Delete only explicit temporary files AFTER the following GRPO save.

    A completion marker alone is not enough: replay after a pre-checkpoint
    crash still needs draft_fresh/stale. Scalar journals and raw data survive.
    """
    removed = []
    if keep:
        return removed
    for marker in Path(output_dir).glob("boundaries/step_*/complete.json"):
        completion = json.loads(marker.read_text(encoding="utf-8"))
        if int(completion["next_target_optimizer_step"]) > durable_target_step:
            continue
        for name in ("phi_base.pt", "draft_stale.pt", "draft_fresh.pt"):
            path = marker.parent / name
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return removed


def mark_durable_boundaries(output_dir, target_step):
    """The training checkpoint is committed before its result markers."""
    for marker in Path(output_dir).glob('boundaries/step_*/complete.json'):
        completion = json.loads(marker.read_text(encoding='utf-8'))
        if int(completion['next_target_optimizer_step']) <= target_step and not completion.get('durable_target_checkpoint'):
            completion['durable_target_checkpoint'] = True
            atomic_json(marker, completion)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def load_completed_results(output_dir: Path) -> tuple[set[int], list[dict], list[BranchSummary]]:
    """Resume only boundaries with a completion marker, never partial exports."""
    protocol_path = output_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
    completed = set()
    for marker in output_dir.glob("boundaries/step_*/complete.json"):
        payload = json.loads(marker.read_text(encoding="utf-8"))
        step = int(payload["policy_step"])
        if marker.parent.name != f"step_{step}":
            raise ValueError(f"completion marker has mismatched policy_step: {marker}")
        if protocol.get('format') == PROTOCOL_VERSION and not payload.get('durable_target_checkpoint'):
            continue
        completed.add(step)

    def read_jsonl(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    if protocol.get("format") == PROTOCOL_VERSION:
        # Canonical small per-boundary journals recover even an interrupted
        # append of the global CSV/JSONL files, with no duplicate rows.
        responses, summaries = [], []
        for step in sorted(completed):
            journal = output_dir / "boundaries" / f"step_{step}" / "results.json"
            if not journal.is_file():
                raise RuntimeError(f"completed boundary has no result journal: {journal}")
            payload = json.loads(journal.read_text(encoding="utf-8"))
            if int(payload['policy_step']) != step:
                raise RuntimeError(f"result journal policy_step mismatch: {journal}")
            responses.extend(payload['per_response'])
            boundary_summaries = [BranchSummary(**row) for row in payload['summaries']]
            timing_path = journal.parent / 'io_timing.json'
            if timing_path.is_file():
                io_ms = json.loads(timing_path.read_text())['analysis_io_wall_ms']
                boundary_summaries = [replace(row, analysis_io_wall_ms=io_ms) for row in boundary_summaries]
            summaries.extend(boundary_summaries)
    elif completed and not all(
        (output_dir / name).is_file() for name in ("per_response.jsonl", "summary.jsonl")
    ):
        raise RuntimeError(f"completed policy-lag boundaries lack result files in {output_dir}")
    if protocol.get("format") != PROTOCOL_VERSION:
        responses = [row for row in read_jsonl(output_dir / "per_response.jsonl")
                     if int(row["policy_step"]) in completed]
        summaries = [BranchSummary(**row) for row in read_jsonl(output_dir / "summary.jsonl")
                     if int(row["policy_step"]) in completed]
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("format") in {
            "fastgrpo_policy_lag_protocol_v3", "fastgrpo_policy_lag_protocol_v4"
        } or protocol.get("format") == PROTOCOL_VERSION:
            required = (set(BRANCHES) if protocol.get("format") == PROTOCOL_VERSION
                        else {"base", "stale", "fresh"})
            for step in completed:
                response_branches = {row["branch"] for row in responses if int(row["policy_step"]) == step}
                step_summaries = [row for row in summaries if row.policy_step == step]
                summary_branches = {row.branch for row in step_summaries}
                if response_branches != required or summary_branches != required:
                    raise RuntimeError(
                        f"completed policy-lag boundary {step} is missing paired "
                        f"{'/'.join(sorted(required))} records in {output_dir}"
                    )
                if len(step_summaries) != len(required):
                    raise RuntimeError(
                        f"completed policy-lag boundary {step} has duplicate branch summaries in {output_dir}"
                    )
                if protocol.get('format') == PROTOCOL_VERSION:
                    step_rows = [row for row in responses if int(row['policy_step']) == step]
                    for branch in ('fresh', 'reflex'):
                        bootstrap_delta_by_prompt(
                            [row for row in step_rows if row['branch'] == 'stale'],
                            [row for row in step_rows if row['branch'] == branch], seed=0, samples=1)
    return completed, responses, summaries


def append_jsonl(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a non-empty comma-separated integer list")
    return values


def state_digest(state: Mapping) -> str:
    """Stable identity check for cloned model/optimizer states."""
    import torch

    digest = hashlib.sha256()

    def visit(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().resolve_conj().resolve_neg().contiguous()
            if tensor.layout != torch.strided:
                raise TypeError(
                    f"state_digest supports dense model/optimizer tensors, got {tensor.layout}"
                )
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            # A dtype-changing view cannot operate directly on a scalar tensor
            # in current PyTorch releases. Flatten first so 0-D AdamW ``step``
            # tensors, BF16 parameters and empty tensors all share the same
            # byte-oriented hashing path.
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, Mapping):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())

    visit(state)
    return digest.hexdigest()


def weighted_aal(records: Sequence[Mapping]) -> tuple[float, int, int, int]:
    """FastGRPO AAL: sum accepted lengths / sequence verification rounds.

    ``accepted_length_sum`` is FastGRPO ``total_acc_length`` and includes the
    verified root/bonus target token. ``verification_rounds`` is
    ``total_decoded_token_num``. This is intentionally not a mean of batch AALs.
    """
    accepted = sum(int(row["accepted_length_sum"]) for row in records)
    rounds = sum(int(row["verification_rounds"]) for row in records)
    generated = sum(int(row.get("generated_tokens", 0)) for row in records)
    if rounds <= 0:
        raise ValueError("AAL requires at least one sequence verification round")
    if accepted < rounds:
        raise ValueError("accepted length cannot be smaller than verification rounds")
    return accepted / rounds, accepted, rounds, generated


def bootstrap_delta_by_prompt(
    stale: Sequence[Mapping], fresh: Sequence[Mapping], *, seed: int, samples: int = 2000
) -> dict:
    """Prompt-cluster bootstrap; responses/seeds within a prompt stay grouped."""
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    by_branch = {}
    for name, rows in (("stale", stale), ("fresh", fresh)):
        grouped = {}
        for row in rows:
            grouped.setdefault(str(row["prompt_id"]), []).append(row)
        by_branch[name] = grouped
    stale_ids = set(by_branch["stale"])
    fresh_ids = set(by_branch["fresh"])
    if stale_ids != fresh_ids:
        raise ValueError(
            "stale/fresh rollout prompt sets differ: "
            f"missing from fresh={sorted(stale_ids - fresh_ids)}, "
            f"missing from stale={sorted(fresh_ids - stale_ids)}"
        )
    prompt_ids = sorted(stale_ids)
    if not prompt_ids:
        raise ValueError("stale/fresh evaluation has no common prompt_id")
    for prompt_id in prompt_ids:
        stale_rows = by_branch["stale"][prompt_id]
        fresh_rows = by_branch["fresh"][prompt_id]
        if len(stale_rows) != len(fresh_rows):
            raise ValueError(f"stale/fresh response counts differ for prompt_id={prompt_id}")
        if all("response_index" in row for row in stale_rows + fresh_rows):
            stale_indices = sorted(int(row["response_index"]) for row in stale_rows)
            fresh_indices = sorted(int(row["response_index"]) for row in fresh_rows)
            if stale_indices != fresh_indices:
                raise ValueError(f"stale/fresh response indices differ for prompt_id={prompt_id}")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        sampled = rng.choice(prompt_ids, size=len(prompt_ids), replace=True)
        s_rows, f_rows = [], []
        for prompt_id in sampled:
            s_rows.extend(by_branch["stale"][str(prompt_id)])
            f_rows.extend(by_branch["fresh"][str(prompt_id)])
        deltas.append(weighted_aal(f_rows)[0] - weighted_aal(s_rows)[0])
    point = weighted_aal(fresh)[0] - weighted_aal(stale)[0]
    low, high = np.quantile(np.asarray(deltas), [0.025, 0.975])
    return {
        "delta_aal": float(point),
        "delta_aal_ci_low": float(low),
        "delta_aal_ci_high": float(high),
        "bootstrap_unit": "prompt",
        "bootstrap_samples": int(samples),
    }


def teacher_shift_tv(logits_t, logits_t1, valid_mask=None, row_chunk_size: int = 32) -> float:
    """Exact mean full-vocabulary TV in FP32 at temperature 1, before filtering."""
    import torch

    if logits_t.shape != logits_t1.shape:
        raise ValueError(f"teacher logits shape mismatch: {logits_t.shape} vs {logits_t1.shape}")
    left = logits_t.reshape(-1, logits_t.shape[-1])
    right = logits_t1.reshape(-1, logits_t1.shape[-1])
    mask = (
        torch.ones(left.shape[0], dtype=torch.bool, device=left.device)
        if valid_mask is None
        else valid_mask.reshape(-1).to(device=left.device, dtype=torch.bool)
    )
    total = torch.zeros((), dtype=torch.float64, device=left.device)
    count = 0
    for start in range(0, left.shape[0], row_chunk_size):
        chosen = mask[start : start + row_chunk_size]
        if not chosen.any():
            continue
        p = torch.softmax(left[start : start + row_chunk_size][chosen].float(), dim=-1)
        q = torch.softmax(right[start : start + row_chunk_size][chosen].float(), dim=-1)
        total += (0.5 * torch.abs(p - q).sum(-1)).double().sum()
        count += int(chosen.sum().item())
    if count == 0:
        raise ValueError("teacher_shift_tv has no valid prefix positions")
    return float((total / count).cpu())


@dataclass
class BranchSummary:
    policy_step: int
    seed: int
    branch: str
    aal: float
    delta_aal: float | None
    verification_rounds: int
    generated_tokens: int
    actual_training_token_count: int
    optimizer_steps: int
    policy_checkpoint_id: str
    draft_checkpoint_id: str
    feature_policy_version: str
    teacher_shift_tv: float
    ci_low: float | None = None
    ci_high: float | None = None
    delta_vs_base: float | None = None
    evaluation_epoch: int | None = None
    evaluation_batch: int | None = None
    used_for_grpo: bool | None = None
    evaluation_prompt_batch_id: str | None = None
    requested_training_token_budget: int | None = None
    effective_draft_lr: float | None = None
    accepted_length_sum: int | None = None
    online_draft_update_gpu_ms: float | None = None
    reflex_update_gpu_ms: float | None = None
    reflex_update_count: int | None = None
    reflex_update_gpu_ms_per_update: float | None = None
    reflex_update_exposed_ms: float | None = None
    analysis_io_wall_ms: float | None = None


class BoundaryJournal:
    """Small resumable state machine; completed stages are never repeated."""

    STAGES = ("saved_base", "collected_stale", "updated_policy", "collected_fresh", "trained", "evaluated", "exported")

    def __init__(self, output_dir: Path, policy_step: int):
        self.path = output_dir / "boundaries" / f"step_{policy_step}" / "journal.json"
        self.payload = {"policy_step": policy_step, "completed": [], "artifacts": {}}
        if self.path.is_file():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))

    def done(self, stage: str) -> bool:
        return stage in self.payload["completed"]

    def complete(self, stage: str, **artifacts) -> None:
        if stage not in self.STAGES:
            raise ValueError(f"unknown boundary stage: {stage}")
        if stage not in self.payload["completed"]:
            expected = self.STAGES[len(self.payload["completed"])]
            if stage != expected:
                raise RuntimeError(f"boundary stage order violation: expected {expected}, got {stage}")
            self.payload["completed"].append(stage)
        self.payload["artifacts"].update(artifacts)
        atomic_json(self.path, self.payload)


def write_results(
    output_dir: Path,
    per_response: Sequence[Mapping],
    summaries: Sequence[BranchSummary],
    *, plot: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "per_response.jsonl"
    atomic_text(
        response_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in per_response),
    )
    rows = [asdict(item) for item in summaries]
    summary_jsonl = output_dir / "summary.jsonl"
    atomic_text(
        summary_jsonl,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(BranchSummary.__dataclass_fields__))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output_dir / "summary.csv", stream.getvalue())
    step_rows = build_step_metrics(per_response, rows)
    for row in step_rows:
        timing_path = output_dir / 'boundaries' / f"step_{row['policy_step']}" / 'io_timing.json'
        if timing_path.is_file():
            row['analysis_io_wall_ms'] = json.loads(timing_path.read_text())['analysis_io_wall_ms']
    atomic_text(output_dir / "step_metrics.jsonl", "".join(json.dumps(row) + "\n" for row in step_rows))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=STEP_METRIC_COLUMNS)
    writer.writeheader()
    writer.writerows(step_rows)
    atomic_text(output_dir / "step_metrics.csv", stream.getvalue())
    if not plot:
        return {"status": "deferred", "reason": "plot only at end of run or --mode plot"}
    plot_status = plot_results(rows, output_dir / "aal_policy_lag.png")
    atomic_json(output_dir / "plot_status.json", plot_status)
    if plot_status["status"] == "skipped":
        warnings.warn(
            "Optional policy-lag plot was skipped; JSONL/CSV results were exported. "
            f"Reason: {plot_status['reason']}",
            RuntimeWarning,
            stacklevel=2,
        )
    return plot_status


def build_step_metrics(per_response, summaries):
    """One exact ratio-of-sums row per three-way boundary, never mean AAL."""
    output = []
    for step in sorted({int(row['policy_step']) for row in summaries}):
        by_branch = {row['branch']: row for row in summaries if int(row['policy_step']) == step}
        if not set(BRANCHES) <= set(by_branch):
            continue  # Old v3/v4 results remain exportable/plotable.
        if sum(int(row['policy_step']) == step for row in summaries) != 3:
            raise ValueError(f"duplicate or unexpected branch summary at boundary {step}")
        row = dict.fromkeys(STEP_METRIC_COLUMNS)
        row['policy_step'] = step
        for branch in BRANCHES:
            records = [item for item in per_response
                       if int(item['policy_step']) == step and item['branch'] == branch]
            aal, accepted, rounds, _ = weighted_aal(records)
            row[f'{branch}_aal'] = aal
            row[f'{branch}_accepted_length_sum'] = accepted
            row[f'{branch}_verification_rounds'] = rounds
            if not np.isclose(aal, by_branch[branch]['aal']):
                raise ValueError(f"summary AAL disagrees with raw counters: {step}/{branch}")
        for branch in ('fresh', 'reflex'):
            row[f'{branch}_minus_stale_aal'] = row[f'{branch}_aal'] - row['stale_aal']
            row[f'{branch}_minus_stale_ci_low'] = by_branch[branch].get('ci_low')
            row[f'{branch}_minus_stale_ci_high'] = by_branch[branch].get('ci_high')
        stale, fresh, reflex = (by_branch[branch] for branch in BRANCHES)
        row.update(teacher_shift_tv=stale['teacher_shift_tv'],
                   requested_training_token_budget=stale.get('requested_training_token_budget'),
                   actual_training_token_budget=stale['actual_training_token_count'],
                   draft_optimizer_steps=stale['optimizer_steps'], effective_draft_lr=stale.get('effective_draft_lr'),
                   stale_online_draft_update_gpu_ms=stale.get('online_draft_update_gpu_ms'),
                   fresh_online_draft_update_gpu_ms=fresh.get('online_draft_update_gpu_ms'),
                   analysis_io_wall_ms=stale.get('analysis_io_wall_ms'))
        for name in ('reflex_update_gpu_ms', 'reflex_update_count', 'reflex_update_gpu_ms_per_update',
                     'reflex_update_exposed_ms'):
            row[name] = reflex.get(name)
        output.append(row)
    return output


def _append_csv(path, rows, columns):
    has_header = path.is_file() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        if not has_header:
            writer.writeheader()
        writer.writerows(rows)


def append_boundary_results(output_dir, per_response, summaries, completion, *, io_meter):
    """Append only this boundary; journal first, completion marker last.

    Global append files are derived views. Startup/end aggregation rebuilds
    them once from complete journals if a process died mid-append.
    """
    step = int(completion['policy_step'])
    boundary = output_dir / 'boundaries' / f'step_{step}'
    before = io_meter.wall_ms
    rows = [asdict(item) for item in summaries]
    metrics = build_step_metrics(per_response, rows)[0]
    with io_meter.measure():
        atomic_json(boundary / 'results.json', dict(policy_step=step, per_response=per_response, summaries=rows))
        for filename, records in (('per_response.jsonl', per_response), ('summary.jsonl', rows)):
            with (output_dir / filename).open('a', encoding='utf-8') as stream:
                stream.write(''.join(json.dumps(row, sort_keys=True) + '\n' for row in records))
        _append_csv(output_dir / 'summary.csv', rows, list(rows[0]))
    metrics['analysis_io_wall_ms'] = float(completion.get('analysis_io_wall_ms', 0)) + io_meter.wall_ms - before
    with io_meter.measure():
        append_jsonl(output_dir / 'step_metrics.jsonl', metrics)
        _append_csv(output_dir / 'step_metrics.csv', [metrics], STEP_METRIC_COLUMNS)
        atomic_json(boundary / 'complete.json', completion)
    metrics['analysis_io_wall_ms'] = float(completion.get('analysis_io_wall_ms', 0)) + io_meter.wall_ms - before
    atomic_json(boundary / 'io_timing.json', {'analysis_io_wall_ms': metrics['analysis_io_wall_ms']})
    return metrics


def plot_results(rows: Sequence[Mapping], path: Path) -> dict:
    """Best-effort rendering of the derived policy-lag plot."""
    if not rows:
        return {"status": "skipped", "reason": "no summary rows", "path": str(path)}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Skipping optional policy-lag plot: {exc}", RuntimeWarning, stacklevel=2)
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "path": str(path)}
    figure = None
    try:
        grouped = {}
        for row in rows:
            grouped.setdefault((int(row["policy_step"]), row["branch"]), []).append(float(row["aal"]))
        steps = sorted({key[0] for key in grouped})
        figure, axis = plt.subplots(figsize=(9, 5))
        for branch in BRANCHES:
            if all((step, branch) in grouped for step in steps):
                if any(len(grouped[(step, branch)]) != 1 for step in steps):
                    raise ValueError('plot requires one aggregate branch AAL per policy step')
                axis.plot(steps, [grouped[(step, branch)][0] for step in steps], label=branch)
        axis.set_ylabel("AAL (root/bonus included)")
        axis.legend()
        axis.set_xlabel("policy step (evaluated on next real GRPO rollout)")
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        figure.savefig(temporary, format="png", dpi=180)
        os.replace(temporary, path)
    except Exception as exc:
        warnings.warn(f"Skipping optional policy-lag plot: {exc}", RuntimeWarning, stacklevel=2)
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "path": str(path)}
    finally:
        if figure is not None:
            plt.close(figure)
    return {"status": "generated", "path": str(path)}


def validate_paths(args) -> None:
    required_files = {
        "target model config": Path(args.model_dir) / "config.json",
        "draft config": Path(args.draft_config),
        "dataset": Path(args.dataset_path),
    }
    if args.draft_initialization_mode == "pretrained":
        required_files["draft checkpoint"] = Path(args.draft_checkpoint)
    if args.target_adapter:
        required_files["target adapter/checkpoint"] = Path(args.target_adapter)
    if args.vocab_mapping:
        required_files["vocabulary mapping"] = Path(args.vocab_mapping)
    if args.eval_dataset_path:
        required_files["evaluation dataset"] = Path(args.eval_dataset_path)
    failures = [f"{name}: {path}" for name, path in required_files.items() if not path.exists()]
    if failures:
        raise FileNotFoundError("missing required paths:\n  " + "\n  ".join(failures))


def dataset_disjointness_report(train_path: Path, eval_path: Path) -> dict:
    """Check the prepared split by source ID and prompt content, not filenames."""
    def read_keys(path):
        source_ids, prompt_hashes, count = set(), set(), 0
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                count += 1
                if "source_index" in row:
                    source_ids.add(int(row["source_index"]))
                prompt = row.get("question", row.get("prompt"))
                if prompt is None:
                    raise ValueError(f"dataset row {count} has no prompt in {path}")
                prompt_hashes.add(hashlib.sha256(str(prompt).encode("utf-8")).hexdigest())
        return source_ids, prompt_hashes, count

    train_ids, train_prompts, train_count = read_keys(train_path)
    eval_ids, eval_prompts, eval_count = read_keys(eval_path)
    source_overlap = len(train_ids & eval_ids)
    prompt_overlap = len(train_prompts & eval_prompts)
    if source_overlap:
        raise ValueError(f"train/eval source indices overlap: {source_overlap}")
    return {
        "train_path": str(train_path.resolve()),
        "eval_path": str(eval_path.resolve()),
        "train_rows": train_count,
        "eval_rows": eval_count,
        "source_index_overlap": source_overlap,
        "prompt_overlap": prompt_overlap,
        "source_disjoint": source_overlap == 0,
        "prompt_content_disjoint": prompt_overlap == 0,
        "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        "evaluation_usage": "split_integrity_only; AAL uses next real GRPO rollout",
    }


def dependency_report(repo: Path) -> dict:
    report = {
        "fastgrpo_expected_commit": FASTGRPO_COMMIT,
        "specforge_expected_commit": SPECFORGE_COMMIT,
        "python": sys.version.split()[0],
    }
    try:
        report["fastgrpo_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception as exc:
        report["fastgrpo_commit_error"] = str(exc)
    source_files = (
        "grpo_speculative.py", "helper/specualtive_generate.py",
        "helper/eagle3_specforge.py", "helper/eagle3_supervision.py",
        "helper/response_batches.py", "helper/rollout_merge.py",
        "helper/policy_lag_protocol.py", "helper/fast_lk_reflex.py",
        "helper/fast_lk_reflex_kernels.py", "helper/tree_verification.py",
        "helper/sampling.py", "helper/rollout_history.py", "helper/reflex_port.json",
        "policy_lag_analysis.py",
        "run_policy_lag_analysis.sh", "run_policy_lag_analysis_b200.sh",
        "scripts/prepare_dapo_policy_lag.py",
        "third_party/SpecForge/specforge/modeling/target/target_head.py",
        "third_party/SpecForge/specforge/algorithms/eagle3/model.py",
    )
    report["source_sha256"] = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in source_files
    }
    report['reflex_port'] = json.loads((repo / 'helper/reflex_port.json').read_text())
    for module in ("torch", "transformers", "peft", "datasets", "specforge", "triton"):
        try:
            imported = __import__(module)
            report[module] = getattr(imported, "__version__", "installed")
        except Exception as exc:
            report[module] = f"MISSING: {type(exc).__name__}: {exc}"
    try:
        import specforge
        package_path = Path(specforge.__file__).resolve()
        vendored_candidates = [
            parent / 'VENDORED_COMMIT'
            for parent in package_path.parents
            if (parent / 'VENDORED_COMMIT').is_file()
        ]
        if vendored_candidates:
            vendored = vendored_candidates[0]
            fields = dict(
                line.split('=', 1)
                for line in vendored.read_text(encoding='utf-8').splitlines()
                if '=' in line
            )
            report['specforge_commit'] = fields.get('commit')
            report['specforge_source'] = str(vendored.parent)
        else:
            try:
                checkout = next(parent for parent in package_path.parents if (parent / '.git').exists())
                report['specforge_commit'] = subprocess.check_output(
                    ['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True
                ).strip()
            except (StopIteration, subprocess.SubprocessError):
                from importlib.metadata import distribution

                direct_url = distribution('specforge').read_text('direct_url.json')
                metadata = json.loads(direct_url or '{}')
                report['specforge_commit'] = metadata.get('vcs_info', {}).get('commit_id')
    except Exception as exc:
        report['specforge_commit_error'] = str(exc)
    return report


def smoke_test(output_dir: Path) -> None:
    """CPU plumbing test: collect -> equal branch train -> evaluate -> export."""
    import torch

    torch.manual_seed(11)
    base = torch.nn.Linear(3, 2, bias=False)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    x_stale = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    x_fresh = torch.tensor([[1.0, 0.1, 1.0], [0.0, 1.1, 1.0]])
    branches = {}
    for name, features in (("stale", x_stale), ("fresh", x_fresh)):
        branch = torch.nn.Linear(3, 2, bias=False)
        branch.load_state_dict(base_state)
        optimizer = torch.optim.AdamW(branch.parameters(), lr=1e-2)
        optimizer.zero_grad()
        # This is only a state/invariant smoke objective, never an experiment
        # result and never presented as SpecForge training.
        branch(features).square().mean().backward()
        optimizer.step()
        branches[name] = (branch, optimizer)
    assert state_digest(base_state) == state_digest({k: v.detach() for k, v in base_state.items()})
    # AdamW stores its step counter as a scalar tensor. Keep this in the smoke
    # test because policy-lag branch restoration hashes the full optimizer
    # state before doing any expensive evaluation rollout.
    optimizer_state = branches["stale"][1].state_dict()
    assert state_digest(optimizer_state) == state_digest(optimizer_state)
    assert len(state_digest({"scalar": torch.tensor(1.0), "bf16": torch.ones(2, dtype=torch.bfloat16)})) == 64
    stale_records = [
        {"prompt_id": "p0", "accepted_length_sum": 5, "verification_rounds": 2, "generated_tokens": 5},
        {"prompt_id": "p1", "accepted_length_sum": 3, "verification_rounds": 2, "generated_tokens": 4},
    ]
    fresh_records = [
        {"prompt_id": "p0", "accepted_length_sum": 6, "verification_rounds": 2, "generated_tokens": 5},
        {"prompt_id": "p1", "accepted_length_sum": 4, "verification_rounds": 2, "generated_tokens": 4},
    ]
    from helper.fast_lk_reflex import FastLKReflex
    persistent = state_digest({'draft': branches['stale'][0].state_dict(),
                               'optimizer': branches['stale'][1].state_dict()})
    reflex = FastLKReflex(feature_dim=2, backend='torch')
    reflex.start(2, 2, 3, 'cpu')
    assert torch.count_nonzero(reflex.state) == 0
    mapping = torch.arange(2)
    logits = branches['stale'][0](x_fresh).unsqueeze(1).detach()
    reflex.propose(logits, x_fresh.unsqueeze(1), 2, mapping, root=True)
    reflex.update_from_target_probs(branches['fresh'][0](x_fresh).detach().softmax(-1), mapping)
    reflex_stats = reflex.finish()
    reflex.clear()
    assert reflex.state is None
    assert persistent == state_digest({'draft': branches['stale'][0].state_dict(),
                                       'optimizer': branches['stale'][1].state_dict()})
    # Synthetic counters test export plumbing only, not measured acceptance.
    reflex_records = [dict(row, accepted_length_sum=row['accepted_length_sum'] + 2) for row in stale_records]
    boots = {name: bootstrap_delta_by_prompt(stale_records, records, seed=7, samples=100)
             for name, records in (('fresh', fresh_records), ('reflex', reflex_records))}
    summaries = []
    for branch, records in (("stale", stale_records), ("fresh", fresh_records), ('reflex', reflex_records)):
        aal, accepted, rounds, generated = weighted_aal(records)
        boot = boots.get(branch, {})
        summaries.append(BranchSummary(
            policy_step=1, seed=7, branch=branch, aal=aal,
            delta_aal=boot.get("delta_aal"),
            verification_rounds=rounds, generated_tokens=generated,
            actual_training_token_count=2, optimizer_steps=1,
            policy_checkpoint_id="smoke-theta-1", draft_checkpoint_id=f"smoke-{branch}",
            feature_policy_version="smoke-theta-1", teacher_shift_tv=0.0,
            ci_low=boot.get("delta_aal_ci_low"),
            ci_high=boot.get("delta_aal_ci_high"),
            accepted_length_sum=accepted,
            reflex_update_count=reflex_stats.updates if branch == 'reflex' else None,
        ))
    tagged = []
    for branch, records in (("stale", stale_records), ("fresh", fresh_records), ('reflex', reflex_records)):
        for row in records:
            tagged.append({**row, "policy_step": 1, "seed": 7, "branch": branch, "smoke_test": True})
    atomic_json(output_dir / 'protocol.json', {'format': PROTOCOL_VERSION, 'research_result': False})
    append_boundary_results(output_dir, tagged, summaries,
        {'policy_step': 1, 'next_target_optimizer_step': 2, 'durable_target_checkpoint': True,
         'research_result': False}, io_meter=AnalysisIO())
    atomic_json(output_dir / "smoke_validation.json", {
        "status": "passed", "research_result": False,
        "checks": ["collect", "identical_initialization", "equal_token_budget", "equal_optimizer_steps",
                   "reflex_uses_stale", "reflex_state_reset", "persistent_optimizer_unchanged", "evaluate",
                   "weighted_aal", "three_way_prompt_bootstrap", "append_export"],
    })


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "smoke", "dependencies", "plot"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--target-adapter", default="")
    parser.add_argument("--draft-checkpoint", default="")
    parser.add_argument("--draft-config", default="")
    parser.add_argument("--draft-initialization-mode", choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--vocab-mapping", default="")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--eval-dataset-path", default="")
    parser.add_argument("--reflex-backend", choices=['triton', 'torch', 'auto'], default='triton')
    return parser


def main():
    args = build_parser().parse_args()
    output = Path(args.output_dir)
    if args.mode == "smoke":
        smoke_test(output)
        print(f"Smoke test passed: {output}")
        return
    if args.mode == "plot":
        completed, responses, summaries = load_completed_results(output)
        if not completed or not summaries:
            raise RuntimeError(f"no completed policy-lag results to plot in {output}")
        status = write_results(output, responses, summaries, plot=True)
        print(json.dumps(status, indent=2, sort_keys=True))
        return
    report = dependency_report(Path(__file__).resolve().parent)
    atomic_json(output / "dependencies.json", report)
    if args.mode == "validate":
        validate_paths(args)
        if args.eval_dataset_path:
            report["train_eval_disjointness"] = dataset_disjointness_report(
                Path(args.dataset_path), Path(args.eval_dataset_path)
            )
        report["analysis_protocol"] = PROTOCOL_VERSION
        atomic_json(output / "dependencies.json", report)
        if sys.version_info < (3, 11):
            raise RuntimeError(
                f'pinned SpecForge requires Python >=3.11; found {sys.version.split()[0]}'
            )
        if str(report.get("specforge", "")).startswith("MISSING"):
            raise RuntimeError(report["specforge"])
        if args.reflex_backend == 'triton' and str(report.get('triton', '')).startswith('MISSING'):
            raise RuntimeError('Production Reflex requires Triton: ' + report['triton'])
        from packaging.version import Version

        torch_version = str(report.get('torch', '')).split('+', 1)[0]
        transformers_version = str(report.get('transformers', '')).split('+', 1)[0]
        try:
            torch_parsed = Version(torch_version)
            transformers_parsed = Version(transformers_version)
        except Exception as exc:
            raise RuntimeError(f'cannot parse runtime dependency versions: {exc}') from exc
        if (torch_parsed.major, torch_parsed.minor) not in {(2, 11), (2, 13)}:
            raise RuntimeError(
                'policy-lag EAGLE-3 supports validated torch 2.11.x or the '
                f'upstream 2.13.x lock; found {report.get("torch")}'
            )
        if not (Version('5.8.0') <= transformers_parsed < Version('6.0.0')):
            raise RuntimeError(
                'policy-lag EAGLE-3 requires transformers >=5.8,<6.0; found '
                f'{report.get("transformers")}'
            )
        if report.get('specforge_commit') != SPECFORGE_COMMIT:
            raise RuntimeError(
                'SpecForge commit mismatch: expected '
                f'{SPECFORGE_COMMIT}, got {report.get("specforge_commit", "unknown")}'
            )
        print(f"Validation passed: {output / 'dependencies.json'}")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

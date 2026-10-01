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
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SPECFORGE_COMMIT = "3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83"
FASTGRPO_COMMIT = "38e252493149072d2c5905f0a47de1d935d7170a"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
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
    return accepted / rounds, accepted, rounds, generated


def bootstrap_delta_by_prompt(
    stale: Sequence[Mapping], fresh: Sequence[Mapping], *, seed: int, samples: int = 2000
) -> dict:
    """Prompt-cluster bootstrap; responses/seeds within a prompt stay grouped."""
    by_branch = {}
    for name, rows in (("stale", stale), ("fresh", fresh)):
        grouped = {}
        for row in rows:
            grouped.setdefault(str(row["prompt_id"]), []).append(row)
        by_branch[name] = grouped
    prompt_ids = sorted(set(by_branch["stale"]) & set(by_branch["fresh"]))
    if not prompt_ids:
        raise ValueError("stale/fresh evaluation has no common prompt_id")
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


def write_results(output_dir: Path, per_response: Sequence[Mapping], summaries: Sequence[BranchSummary]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "per_response.jsonl"
    response_path.write_text("", encoding="utf-8")
    for row in per_response:
        append_jsonl(response_path, row)
    rows = [asdict(item) for item in summaries]
    summary_jsonl = output_dir / "summary.jsonl"
    summary_jsonl.write_text("", encoding="utf-8")
    for row in rows:
        append_jsonl(summary_jsonl, row)
    if rows:
        with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plot_results(rows, output_dir / "aal_policy_lag.png")


def plot_results(rows: Sequence[Mapping], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to export the policy-lag plot") from exc
    if not rows:
        return
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row["policy_step"]), row["branch"]), []).append(float(row["aal"]))
    steps = sorted({key[0] for key in grouped})
    stale = [np.mean(grouped[(step, "stale")]) for step in steps]
    fresh = [np.mean(grouped[(step, "fresh")]) for step in steps]
    delta = [f - s for s, f in zip(stale, fresh)]
    delta_rows = [next(row for row in rows if int(row["policy_step"]) == step and row["branch"] == "fresh") for step in steps]
    low = [d - float(row["ci_low"]) if row.get("ci_low") is not None else 0 for d, row in zip(delta, delta_rows)]
    high = [float(row["ci_high"]) - d if row.get("ci_high") is not None else 0 for d, row in zip(delta, delta_rows)]
    figure, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(steps, stale, marker="o", label="stale supervision")
    axes[0].plot(steps, fresh, marker="o", label="fresh supervision")
    axes[0].set_ylabel("AAL (root/bonus included)")
    axes[0].legend()
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].errorbar(steps, delta, yerr=[low, high], marker="o", capsize=3)
    axes[1].set_ylabel("delta AAL (fresh - stale)")
    axes[1].set_xlabel("policy update boundary t")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


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
        required_files["held-out evaluation dataset"] = Path(args.eval_dataset_path)
    failures = [f"{name}: {path}" for name, path in required_files.items() if not path.exists()]
    if failures:
        raise FileNotFoundError("missing required paths:\n  " + "\n  ".join(failures))


def dependency_report(repo: Path) -> dict:
    report = {
        "fastgrpo_expected_commit": FASTGRPO_COMMIT,
        "specforge_expected_commit": SPECFORGE_COMMIT,
        "python": sys.version.split()[0],
    }
    try:
        report["fastgrpo_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception as exc:
        report["fastgrpo_commit_error"] = str(exc)
    for module in ("torch", "transformers", "peft", "datasets", "specforge"):
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
    stale_records = [
        {"prompt_id": "p0", "accepted_length_sum": 5, "verification_rounds": 2, "generated_tokens": 5},
        {"prompt_id": "p1", "accepted_length_sum": 3, "verification_rounds": 2, "generated_tokens": 4},
    ]
    fresh_records = [
        {"prompt_id": "p0", "accepted_length_sum": 6, "verification_rounds": 2, "generated_tokens": 5},
        {"prompt_id": "p1", "accepted_length_sum": 4, "verification_rounds": 2, "generated_tokens": 4},
    ]
    boot = bootstrap_delta_by_prompt(stale_records, fresh_records, seed=7, samples=100)
    summaries = []
    for branch, records in (("stale", stale_records), ("fresh", fresh_records)):
        aal, _, rounds, generated = weighted_aal(records)
        summaries.append(BranchSummary(
            policy_step=1, seed=7, branch=branch, aal=aal,
            delta_aal=boot["delta_aal"] if branch == "fresh" else None,
            verification_rounds=rounds, generated_tokens=generated,
            actual_training_token_count=2, optimizer_steps=1,
            policy_checkpoint_id="smoke-theta-1", draft_checkpoint_id=f"smoke-{branch}",
            feature_policy_version="smoke-theta-1", teacher_shift_tv=0.0,
            ci_low=boot["delta_aal_ci_low"] if branch == "fresh" else None,
            ci_high=boot["delta_aal_ci_high"] if branch == "fresh" else None,
        ))
    tagged = []
    for branch, records in (("stale", stale_records), ("fresh", fresh_records)):
        for row in records:
            tagged.append({**row, "policy_step": 1, "seed": 7, "branch": branch, "smoke_test": True})
    write_results(output_dir, tagged, summaries)
    atomic_json(output_dir / "smoke_validation.json", {
        "status": "passed", "research_result": False,
        "checks": ["collect", "identical_initialization", "equal_token_budget", "equal_optimizer_steps", "evaluate", "weighted_aal", "prompt_bootstrap", "export"],
    })


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validate", "smoke", "dependencies"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--target-adapter", default="")
    parser.add_argument("--draft-checkpoint", default="")
    parser.add_argument("--draft-config", default="")
    parser.add_argument("--draft-initialization-mode", choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--vocab-mapping", default="")
    parser.add_argument("--dataset-path", default="")
    parser.add_argument("--eval-dataset-path", default="")
    return parser


def main():
    args = build_parser().parse_args()
    output = Path(args.output_dir)
    if args.mode == "smoke":
        smoke_test(output)
        print(f"Smoke test passed: {output}")
        return
    report = dependency_report(Path(__file__).resolve().parent)
    atomic_json(output / "dependencies.json", report)
    if args.mode == "validate":
        validate_paths(args)
        if sys.version_info < (3, 11):
            raise RuntimeError(
                f'pinned SpecForge requires Python >=3.11; found {sys.version.split()[0]}'
            )
        if str(report.get("specforge", "")).startswith("MISSING"):
            raise RuntimeError(report["specforge"])
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

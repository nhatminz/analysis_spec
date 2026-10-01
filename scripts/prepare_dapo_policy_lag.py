#!/usr/bin/env python3
"""Create deterministic, disjoint DAPO train/eval JSONL files.

The source DAPO English parquet exposes only a ``train`` split.  Policy-lag
evaluation must be held out, so this utility selects the requested GRPO pool
and then selects evaluation prompts from the remaining shuffled indices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from datasets import load_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _answer(row: dict) -> str:
    reward = row.get("reward_model")
    answer = reward.get("ground_truth") if isinstance(reward, dict) else row.get("answer")
    if answer is None:
        raise ValueError("DAPO row has no reward_model.ground_truth or answer")
    value = str(answer)
    return value if "\\boxed{" in value else f"\\boxed{{{value}}}"


def _write_jsonl(path: Path, dataset, indices: list[int]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for source_index in indices:
            row = dataset[source_index]
            prompt = row.get("prompt")
            if isinstance(prompt, list):
                prompt = next(
                    (item.get("content") for item in reversed(prompt) if item.get("role") == "user"),
                    prompt[0].get("content") if prompt else None,
                )
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"DAPO row {source_index} has no usable prompt")
            record = {
                "question": prompt,
                "answer": _answer(row),
                "source_index": int(source_index),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = Path(args.input_parquet).resolve()
    output = Path(args.output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"DAPO parquet not found: {source}")
    if args.train_samples <= 0 or args.eval_samples <= 0:
        raise ValueError("train-samples and eval-samples must both be positive")
    output.mkdir(parents=True, exist_ok=True)
    train_path = output / "train.jsonl"
    eval_path = output / "eval.jsonl"
    manifest_path = output / "split_manifest.json"
    requested = {
        "format": "fastgrpo_dapo_policy_lag_split_v1",
        "source_path": str(source),
        "source_sha256": _sha256(source),
        "seed": int(args.seed),
        "train_samples": int(args.train_samples),
        "eval_samples": int(args.eval_samples),
    }
    if manifest_path.is_file() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: existing.get(key) for key in requested}
        if comparable != requested:
            raise RuntimeError(
                f"existing split manifest differs from requested split: {manifest_path}; "
                "use a new output directory or pass --force"
            )
        if not train_path.is_file() or not eval_path.is_file():
            raise RuntimeError(f"split manifest exists but JSONL files are missing in {output}")
        print(f"Reusing deterministic DAPO split at {output}")
        return

    dataset = load_dataset("parquet", data_files=str(source), split="train")
    needed = args.train_samples + args.eval_samples
    if needed > len(dataset):
        raise ValueError(f"requested {needed} rows but source contains only {len(dataset)}")
    indices = list(range(len(dataset)))
    random.Random(args.seed).shuffle(indices)
    train_indices = indices[: args.train_samples]
    eval_indices = indices[args.train_samples : needed]
    if set(train_indices) & set(eval_indices):
        raise AssertionError("internal error: train/eval indices overlap")

    _write_jsonl(train_path, dataset, train_indices)
    _write_jsonl(eval_path, dataset, eval_indices)
    manifest = {
        **requested,
        "source_rows": len(dataset),
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_index_sha256": hashlib.sha256(json.dumps(train_indices).encode()).hexdigest(),
        "eval_index_sha256": hashlib.sha256(json.dumps(eval_indices).encode()).hexdigest(),
        "overlap": 0,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

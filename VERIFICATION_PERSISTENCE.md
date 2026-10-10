# Frozen FastGRPO verification-error persistence

`run_verification_persistence.py` runs a standalone observational experiment. It restores
`target_lora` and the pure `shadow.weights` from an online-trained
`simplelr_opd_policy_lag_a1a2_v2` checkpoint. It creates one frozen draft and target,
without a Reflex projector, optimizer, GRPO rollout, or parameter update.

The existing deterministic SimpleLR subset (`seed=2026`) supplies 64 prompts:
the first 16 are donor responses, and the remaining 48 are evaluation responses.
All use batch size 1, one response, and the original FastGRPO sampling
(`temperature=1`, `top_p=.95`, `top_k=None`), with the existing exact 256-token cap
and EOS behavior. Other tree settings come from the checkpoint's experiment manifest.
The base model weights, checkpoint/source-manifest identity, test file and all 64
rendered/tokenized subset entries are checked before generation.

## Run on B200

Assuming `analysis_spec` is deployed beside `SpecNaacl` under
`/workspace/storage-shared/nlp/minhpn19`, and the online training output used the
standard `outputs/qwen25_3b_simplelr_a1_a2` directory, run in the activated B200
Python environment:

```bash
cd /workspace/storage-shared/nlp/minhpn19/analysis_spec
PYTHON_BIN=python3 \
CUDA_VISIBLE_DEVICES=0 \
bash run_verification_persistence.sh \
  --checkpoint "$PWD/outputs/qwen25_3b_simplelr_a1_a2/checkpoints/latest.pt" \
  --model /workspace/storage-shared/models/Qwen2.5-3B-Instruct \
  --train-path /workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet \
  --test-path /workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet \
  --output-dir "$PWD/outputs/verification_persistence_64" \
  --wall-clock-minutes 150
```

This is the full **64-prompt inference protocol**, not training. If your B200
online training used another output directory (for example `outputs/qwen3b_a1_a2`),
change only `--checkpoint` to that run's `checkpoints/latest.pt`. Use the online
analysis checkpoint containing `target_lora` and `shadow.weights`, rather than
the pretrained draft-only checkpoint in `SpecNaacl`. Its sibling `analysis/manifest.json`
is loaded automatically; use `--source-manifest` if the checkpoint was copied elsewhere.
The checkpoint must remain stable during startup/loading/hashing.

Environment variables supported: `PYTHON_BIN`, `CUDA_VISIBLE_DEVICES`, `MODEL`,
`TRAIN_DATASET_PATH`, `TEST_DATASET_PATH`, `DATA_ROOT`, `OUTPUT_DIR`, and
`ANALYSIS_CHECKPOINT` (alias: `CHECKPOINT`). Without a checkpoint argument, the
launcher uses `outputs/qwen25_3b_simplelr_a1_a2/checkpoints/latest.pt`; it does not
fall back to a local validation checkpoint. Model and data paths otherwise come
from the source manifest. Output directories must be fresh.

For a short execution check, add `--smoke` and use a fresh output directory.
This uses 2 donors, 2 evaluation responses and a 32-token cap. It is explicitly
nonpublishable. No long GRPO training is launched by either command.

## Measurement and missing data

The optional `observation_hook` captures q directly from the draft's root softmax
before top-k proposal construction. The target sampler invokes its observer with
the actual storage-precision probabilities consumed by multinomial, after
temperature/top-p/top-k, without another probability computation or forward pass.
Both vectors refer to the token **after the same root input**. Each verification
checks root input token, logical target position, and prefix cache length, accounting
for FastGRPO's shifted draft input convention. Round indices start at zero.

The hook selects eight strictly positive entries of p−q on the GPU. It transfers
only those token IDs and sparse future means; it never transfers a probability
vector to CPU. Full-vocabulary transient tensors remain on GPU. If fewer than eight
positive entries exist, that origin has no valid set; it is never filled with
nonpositive tokens. Each recipient origin chooses a donor set at exactly the same
round index with a private deterministic RNG, independently of generation RNG.
Both its own set and the donor set are measured on the recipient's context at r+d.
Donors are never chosen from evaluation responses.

`results.csv` contains one row per evaluation origin and distance 1–5, including
missing measurements. Its status distinguishes paired observations, an invalid
origin set, absent donor rounds, donor rounds without a valid positive-eight set,
and unfinished horizons. `control_status` additionally records availability when
the origin itself is invalid. Prefill EOS responses have zero verification rounds
and no origin rows; they remain in response counts and the bootstrap population.
Verified EOS ends the available horizons normally. Missing values are empty CSV
cells / JSON nulls, never synthetic zeros.

## Statistics and artifacts

- `summary.json`: mean error for each distance, paired difference, percentile
  95% confidence intervals from 2,000 resamples of **whole evaluation responses**,
  valid round pairs, contributing responses, control coverage and missing-data counts.
- `results.csv`: sparse per-origin/future measurements, with donor identity.
- `verification_persistence.pdf`: same-response and shuffled-control means with
  confidence intervals, using identical paired support. `same_all` in the summary
  additionally includes valid own-set measurements without an available control.
- `manifest.json`: checkpoint/base-model/tokenizer/source hashes, configuration,
  seeds, implementation hashes, versions, GPU, runtime, freeze audit and sample counts.
- `responses/*.json`: atomic completed-response journals containing only selected
  token sets, root alignment metadata, counts and sparse measurements.

Means weight valid origin rounds equally. Bootstrap clusters include all completed
evaluation responses, even zero-round or zero-valid responses; draws with no valid
measurements are undefined and are omitted with their valid-resample count reported.
Donor sets are conditioned on, not resampled. Control coverage is valid paired
measurements divided by valid own-set measurements with an observed future context.

The 150-minute budget includes startup and is checked **between responses**; a
response already running finishes before the budget check. Completed responses,
CSV, summary and manifest are atomically saved after every response. On budget
exhaustion, interruption or a generation failure, partial results and the PDF are
saved with an explicit status. An over-budget response is never truncated mid-round.
Inference results alone do not establish persistence without sufficient coverage
and evidence in the paired differences/intervals.

## Validation

CPU tests cover token/context alignment, independent same-round donor matching,
recipient-context evaluation, horizon indexing, missing sets, response-cluster
bootstrap computation, zero-round/verified EOS, partial exports and sampler RNG.
CUDA tests compare ordinary and observed FastGRPO under the same seed, including
forced prefill/verified EOS: generated tokens, acceptance counters, RNG state and
model forward counts must match exactly.

```bash
cd /mnt/hdd/nhatminh/SpecDecode/analysis_spec
PYTHONPATH="$PWD/.testdeps:$PWD" /mnt/hdd/nhatminh/fastgrpo_env/bin/python -m pytest -q
```

Actual local smoke timing and counts are recorded in
`validation/verification_persistence_smoke_final/manifest.json` and
`validation/verification_persistence_validation.json`. No full 64-prompt run or
long training was started during implementation.

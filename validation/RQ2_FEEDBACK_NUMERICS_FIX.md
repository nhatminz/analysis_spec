# RQ2 verification-feedback validation fix

The previous RQ2 guard rejected a rollout whenever either
`opd_nonfinite_kl_states` or `opd_invalid_states` was nonzero. These counters
have different meanings. The existing OPD objective deliberately records
infinite probability-space KL separately from its finite `q - p` correction.
Other existing telemetry already leaves the overall KL mean unavailable when
any selected state's KL is nonfinite.

Two real arithmetic cases were reproduced on CPU and CUDA/Triton:

- A draft logit gap of 20 can round its union complement to zero in FP32.
- A gap of 120 underflows non-dominant token probabilities to zero.

Both can produce nonfinite KL with finite probabilities, finite correction
features, and a finite B update. The old guard rejects both. This reproduces
the reported exception, but the user's B200 traceback does not include the
counter values, so it cannot establish which counter triggered that particular
run. The production checkpoint and B200 were unavailable locally.

## Changes

- `rq2/runtime.py` validates actual feedback and the final B scalar diagnostics.
  Nonfinite diagnostic KL alone does not abort. Its count is retained and
  `kl_mean` is `null`, never a mean over just the finite terms.
- `helper/opd_reflex.py` and `helper/opd_reflex_kernels.py` extend
  `opd_invalid_states` to include invalid union probabilities (NaN, Inf or
  negative) and nonfinite correction features, alongside invalid teacher mass.
  CUDA checks run in the existing union kernel and existing scalar packet.
- Actual invalid feedback still aborts with step, rollout, arm, counters, and
  errors. JSON-safe diagnostics are written to the failing arm's `failure.json`
  before raising. No completion marker is written for that arm.
- No epsilon clipping, target/draft forward pass, full-vocabulary CPU copy,
  sampling change, optimizer change, or B-update formula change was introduced.
- Only files inside `analysis_spec/` were edited; `SpecNaacl/` was not edited.

## Validation

CPU environment: Torch 2.13.0+cpu. CUDA environment: Torch 2.6.0+cu124,
Triton 3.2.0, NVIDIA GeForce RTX 3090. Tests use synthetic distributions and a
tiny randomly initialized Qwen model, not the production 3B checkpoint.

| Test suite | Result | Actual pytest runtime |
| --- | --- | --- |
| New numerical regressions + CPU RQ2 + OPD reference tests | 45 passed, 3 CUDA-only skipped | 4.39 s |
| Numerical regressions + existing actual CUDA RQ2 + OPD reference/revision tests | 90 passed | 29.36 s |
| New actual Qwen peaked-draft collection, sync and async | 2 passed | 6.87 s |
| OPD final optimization, KV/cache, and proposal-profile regressions | 61 passed | 3.95 s |

The actual peaked-draft Qwen smoke commits OFF and ON, verifies passive logging
against unobserved replay, preserves RNG/frozen weights, and records nonfinite
KL while keeping B finite. Each arm has 156 nonfinite KL states and zero invalid
feedback states. OFF performs zero updates with zero B norm; ON performs 13
updates. Exact per-arm diagnostics and hashes are in
`rq2_feedback_numerics_validation.json`. These are correctness fixtures, not
scientific evidence for reduced repeated misses or B200 performance.

## B200 rerun, GPU 3

The launcher already runs a smoke pair before full collection and analysis.
Use a new output directory because the code hashes in the previous manifest
must not be overwritten or mixed with corrected runs:

```bash
cd /workspace/storage-shared/nlp/minhpn19/analysis_spec
CUDA_VISIBLE_DEVICES=3 PYTHON_BIN=python3 \
  bash run_rq2_repeated_misses_b200.sh \
  --projector-mode initialized \
  --output-dir "$PWD/outputs/e2_e6_fastgrpo/rq2_repeated_misses_klfix_v1"
```

Keep the existing checkpoint/collection directories. The initialized A remains
an explicitly exploratory condition. Missing proposal calibration and the
Transformers dtype deprecation warning are separate from the fatal guard.

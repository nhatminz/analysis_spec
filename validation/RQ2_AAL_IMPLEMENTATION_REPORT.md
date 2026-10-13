# Frozen step-234 per-prompt RQ2 AAL implementation

Implemented the standalone experiment requested in the attached prompt.
All implementation edits are inside analysis_spec. SpecNaacl source and
checkpoints were only read; no training or training resume was launched.

## Implementation

- `run_rq2_aal_progress.py` provides check/smoke/collect/analyze/all, with the
  required native SpecNaacl resume checkpoint and step 234 defaults.
- `run_rq2_aal_progress_b200.sh` uses B200 model/data paths, offline caches,
  GPU 3 by default, and the existing Python environment.
- `rq2/aal_progress.py` handles learned-A checkpoint decoding, strict CPU
  target/draft restoration before GPU allocation, immutable manifests, frozen
  OFF/ON evaluation, per-arm durable commits, numerical/counter validation,
  passive smoke replay, partial results and evaluation-commit reuse.
- `rq2/per_prompt.py` and `rq2/per_prompt_kernels.py` implement prompt-indexed
  B and prompt-specific feedback normalization. Only group-sensitive kernels
  were ported from read-only SpecNaacl. Existing teacher extraction, union/tail
  objective, proposal normalization and verification kernels are reused.
- `rq2/aal_analysis.py` computes five progress bins, ratio-of-sums AAL within
  each bin, equal-weight paired rollout means and 2,000-resample bootstrap CIs,
  active-response summaries, CSV/JSON and standalone PDF/PNG figures.
- Three shared helpers gained narrow opt-in support: loaded-projector
  installation without initialization, generator prompt-group scope, and a
  scalar group-dispatch count in scheduling. Default batch-wide generation
  retains its behavior. Existing RQ2 runners/statistics and E2/E3/E6 source
  files were not edited; 20 protected source files match analysis_spec HEAD.
- Three new test files cover per-prompt mathematical correctness/isolation,
  weighted denominators, ownership through physical/canonical reorderings,
  sparse/fused/GEMM proposals, native checkpoint restoration, A provenance,
  zero/short rollouts, counter consistency, paired bootstrap and actual CUDA
  frozen generation/logging/EOS/reset behavior.

Full commands, defaults, estimator definitions, output layout and provenance
limitations are in `README_RQ2_AAL_PROGRESS.md`.

## Actual validation

Hardware: NVIDIA GeForce RTX 3090. CUDA environment: Torch 2.6.0+cu124,
Triton 3.2.0. CPU environment: Torch 2.13.0+cpu.

| Run | Result | Actual runtime |
| --- | --- | --- |
| Final CPU new tests + existing RQ2/reviewer/numerics | 54 passed | 15.87 s |
| CUDA new + existing RQ2 and E2E6 inference regression | 47 passed, 3 training-named tests deselected | 28.16 s |
| New prompt-group math/preflight/analysis/CUDA suite | 33 passed | 14.05 s |
| Required production checkpoint preflight | checkpoint unavailable; failed before GPU model allocation | 3.150 s |

The suites overlap; their counts must not be added as independent tests.
The existing reviewer output-scope test requires pytest temporary files to be
inside analysis_spec; the final CPU invocation used
`--basetemp .cache/rq2_aal_cpu_tests_final`. No scope restriction was weakened.

Actual tiny Qwen CUDA evaluation covered synchronous/asynchronous adaptation,
prefill EOS removal, verification EOS swap-removal, all-EOS zero rounds,
unchanged tokens/counters/RNG with passive logging, no extra target forward,
frozen weights, B reset and reuse of committed smoke arms.

An additional actual 8-prompt x 8-response CUDA case verified that prompt-group
OFF matches legacy zero-B tokens and acceptance counters, and that ON updates
all eight prompt groups with no invalid feedback. The two-prompt smoke has
native overall AAL 1.0 in both arms. These tiny random-model fixtures validate
correctness and do not establish any scientific effect on AAL.

Detailed tiny-smoke diagnostics, source hashes, protected-file audit, test
results and production availability are in
`rq2_aal_implementation_validation.json`. The production preflight's actual
stderr is in `rq2_aal_checkpoint_preflight.log`.

## Production availability and interpretation

The workstation does not mount `/workspace/storage-shared/...`; the mandatory
`step234_epoch2_batch0.pt` file is absent. Consequently its SHA256, trained A,
weights and training metadata have **not** been validated locally, and no
production 3B/B200 smoke or 20-rollout experiment was run. No alternative
checkpoint or initialized projector was substituted, and no production
statistics were fabricated. The runner performs the complete preflight on
B200 before any evaluation GPU model is allocated.

If checkpoint B-sharing metadata is absent, provenance explicitly records the
legacy batch-wide training design. The resulting experiment evaluates
per-prompt adaptation on that frozen checkpoint, not a checkpoint trained fully
under per-prompt SPARK. No increasing-AAL hypothesis has been confirmed.

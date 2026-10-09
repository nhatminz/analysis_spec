# Implementation and validation, 2026-10-09

The active experiment is rebuilt entirely inside `analysis_spec`. Every edit made by this implementation is inside that directory. There is no runtime import, symlink or PYTHONPATH dependency on `SpecNaacl`. A user-supplied pretrained checkpoint is read as input data.

Reference integrity matched the before-edit snapshot through the final pilot launch. A later audit observed concurrent external changes to `SpecNaacl/grpo_speculative.py` and the addition of `helper/response_alignment.py` and `helper/shared_adapter.py` (timestamps 15:55–15:56 local time). Those files were only read by this implementation and were preserved. Before/after hashes are in [validation/reference_concurrent_changes.json](validation/reference_concurrent_changes.json). Port provenance remains pinned to the audited original snapshot; none of the copied runtime kernels changed in that reference update. The whole-reference audit correctly reports this difference instead of overwriting its baseline.

## Implemented behavior

`run_policy_lag_motivation.py` and `run_policy_lag_motivation.sh` run one OPD-driven target trajectory with persistent Reflex R/A and independent pure FastGRPO shadow S. A1 Fresh starts from S's exact pre-update state and recomputes both teacher channels on the saved old sequences; it never generates replacement training samples. A2 freezes the same R/A across old/new target and OFF/ON, using the production OPD kernel with LR=0 for OFF. The six conditions share the held-out prompt IDs, decoding settings and per-prompt seeds.

The production defaults are Qwen2.5-3B-Instruct, SimpleLR, 8 unique train prompts × 8 responses per learner, one target optimizer step per iteration, and accumulation=1 for both target and drafts. The reference target accumulation of 4 is deliberately overridden. Measurements are 20/50/100/150/200; 100/200 use 64 test prompts, the others use the first 16 of the immutable 64-list. Step 20 is an explicit zero-update control. Each condition generates one response per test prompt. Length-capped AAL uses actual verification counters and paired prompt-cluster bootstrap.

Data validation found 8,523 train rows, 8,521 unique train questions, 500 unique official test questions and no normalized train/test overlap. The two repeated train rows are recorded and deterministically deduplicated. The 64 test IDs are selected once with seed 2026. Their first two, used by the software smoke, are `test:306` and `test:474`. Prompt-template SHA256 is `9cc66baee1f55efaa786308abe3023e41bac1f12d29fc36a35d907082269e56d`; the ordered subset, question/rendering/token hashes and all duplicates are in `analysis/eval_subset_manifest.json`.

## Changed and retired files

New protocol modules are `motivation/{config,data,state,compatibility,runtime,runner,metrics}.py`, `teacher_relabel.py`, the entrypoint/launcher, `configs/qwen25_3b/simplelr_a1_a2.json`, protocol tests and `scripts/audit_port.py`. Production FastGRPO/Reflex helpers, compatible `train_draft.py`, requirements and licensed source test fixtures are ported locally. Existing policy-lag Python/bash entrypoint names now forward to the single new implementation. README, `DESIGN_A1_A2.md`, `PORT_MANIFEST.md` and `PORT_SOURCES.json` document the implementation.

Active LK/ReflexV5, EAGLE3 policy-lag supervision/runtime, `next_real_grpo_rollout`, DAPO preparation, incompatible model launchers and their old tests/docs are removed or replaced. Unrelated historical SpecForge pretraining tools are preserved under `legacy_pretraining/` with attribution. The complete original path list and before/after hashes are in [PORT_SOURCES.json](PORT_SOURCES.json), with the human-readable mapping in [PORT_MANIFEST.md](PORT_MANIFEST.md).

There are 57 verified source mappings. Critical original source hashes include:

| Reference source | SHA256 |
|---|---|
| `helper/fastgrpo_training.py` | `bbc0971609b7ac0e4961921155ff844faec8de428303fda7e76882fa3b7a9359` |
| `helper/fastgrpo_generate.py` | `dd9495d765b9669a6b673e61e13e2af6f073b4140b00adb1bf42c4bab64efd93` |
| `helper/opd_generate.py` | `d078ad2b23680c83a18e3cbb35ff7c4b2071a7577ed9404b8c9413a4d69da806` |
| `helper/opd_reflex.py` | `cb6041e45aa10109d4280c93f03ffe27fdaac334d2854b23ea4e25a49917e07a` |
| `helper/opd_reflex_kernels.py` | `65fdb32878ec1aec9b3c4c83a105d9f938f7d9512aceed37ed8e2bfefd01259b` |
| `helper/opd_sampling.py` | `04d53d765fe4884aec5403ec4ec1ce869bab314d3fc577e63f36077ebc2bf5e9` |

Only three ported runtime files differ: `get_QAs.py` provides the strict SimpleLR adapter; the two generators add opt-in evaluation caps, and FastGRPO adds opt-in compact teacher tracing. Default training arithmetic remains covered by the source parity tests. Proposal, sampler, feedback, projector optimizer, attention, tree and cache implementations are unchanged copies. Destination hashes and individual test-fixture adaptations are recorded in the machine-readable mapping.

## Exact local inputs

| Input | Path | SHA256 |
|---|---|---|
| Target | `/mnt/hdd/nhatminh/SpecDecode/models/Qwen2.5-3B-Instruct` | config: `eed00b17e22553979d090fa492e587e92885e328914c8e0b0b78f0a0d3576b3b` |
| Model shard 1 | `model-00001-of-00002.safetensors` | `67347b23fb4165b652eb6611f5e1f2a06dfcddba8e909df1b2b0b1857bee06c2` |
| Model shard 2 | `model-00002-of-00002.safetensors` | `a40d941d0e7e0b966ad8b62bb6d6b7c88cce1299197b599d9d0a4ce59aabfc1d` |
| Train | `/mnt/hdd/nhatminh/SpecDecode/data/simplelr_abel_level3to5/train.parquet` | `d4e50e07667d5754f731773dcfd8972003a3ee793f0f4ec8b5b8f52ca0292ee0` |
| Test | `/mnt/hdd/nhatminh/SpecDecode/data/simplelr_abel_level3to5/test.parquet` | `966017e52a78d02d0bd2460b9867f8b562d78c967c0d91218f9a06895bc91e67` |
| Validation-only draft | `/mnt/hdd/nhatminh/.specnaacl-validation/qwen3b-modern/checkpoints/smoke-latest/draft.pth` | `ddadc931fcafe9caeb45ddcc428d8e0912d6aa59a62fe8bc0cfe7c6f8830b278` |
| Draft companion config | same directory, `target_config.json` | `e6822190f2304b583b8ab0c38374f587c770f808ad72538c44cda0b40db6e2a4` |

The draft was supplied through `/mnt/hdd/nhatminh/.specnaacl-validation/qwen3b-modern/latest_checkpoint`, resolving to the file above. This is a **two-step pretrain validation artifact**, used explicitly for software checks. It is never selected as a production default. No checkpoint weights were found inside the supplied `SpecNaacl` tree, including ignored files, and the conventional `SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint` is absent. The main experiment needs the actual pretrained checkpoint from the training machine via `DRAFT_CHECKPOINT`; no retraining or substituted model was performed.

## Executed checks

Commands below are relative to `/mnt/hdd/nhatminh/SpecDecode`.

```bash
# CUDA environment: Python 3.10.20, PyTorch 2.6.0+cu124,
# Transformers 5.12.1, PEFT 0.19.1, RTX 3090 24 GiB.
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
PYTHONPATH="$PWD/analysis_spec:$PWD/analysis_spec/.testdeps" \
TRITON_CACHE_DIR="$PWD/analysis_spec/.cache/triton" \
TORCHINDUCTOR_CACHE_DIR="$PWD/analysis_spec/.cache/inductor" \
/mnt/hdd/nhatminh/fastgrpo_env/bin/python -m pytest -q analysis_spec/tests

# CPU environment: Python 3.12, PyTorch 2.13 CPU.
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONPATH="$PWD/analysis_spec" \
/mnt/hdd/nhatminh/.specnaacl-env-modern/bin/python -m pytest -q analysis_spec/tests

/mnt/hdd/nhatminh/.specnaacl-env-modern/bin/python \
  analysis_spec/scripts/audit_port.py --reference SpecNaacl
```

Final full-suite results: **320 passed, 79 warnings in 20.81s** on CUDA (`validation/cuda_tests_final.log`); **88 passed, 205 skipped in 4.16s** on CPU (`validation/cpu_tests_final.log`). CPU skips cover CUDA/Triton tests; parameterized case counts differ by available backend. The CUDA suite includes source objective/generation/OPD parity, variable-length left-padded teacher replay, both teacher channels, exact response caps, A2 OFF/ON state guards, a nonzero synthetic target update, zero-drift placebo, unscheduled-boundary exclusion, simulated-crash resume and accepted-token/counter equivalence. Target/shadow/RNG equivalence is bitwise; Reflex optimizer atomic rounding is checked at rtol=3e-6, atol=1e-12. The CLI test reopens an existing manifest and rejects a changed configuration.

Additionally, 65 active/test Python files compile, all five active bash launchers parse and CLI help works from `/tmp`. The port audit reported `57 local source mappings verified; reference unchanged` before the concurrent reference update. Local port integrity still passes with `audit_port.py` alone; `--reference SpecNaacl` now correctly flags the three externally changed paths described above. Pytest support packages for the CUDA environment were installed only in ignored `analysis_spec/.testdeps`; no external environment was modified.

The real 3B smoke command is:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
TRITON_CACHE_DIR="$PWD/analysis_spec/.cache/triton" \
TORCHINDUCTOR_CACHE_DIR="$PWD/analysis_spec/.cache/inductor" \
MPLCONFIGDIR="$PWD/analysis_spec/.cache/matplotlib" PYTHONPATH="$PWD/analysis_spec" \
/mnt/hdd/nhatminh/fastgrpo_env/bin/python analysis_spec/run_policy_lag_motivation.py \
  --smoke --train-steps 1 --batch-size 1 --responses-per-prompt 2 \
  --max-length 192 --max-prompt-length 191 --eval-max-new-tokens 4 \
  --bootstrap-samples 100 \
  --draft-checkpoint /mnt/hdd/nhatminh/.specnaacl-validation/qwen3b-modern/latest_checkpoint \
  --output-dir analysis_spec/validation/qwen3b_final_smoke
# Resume: repeat exactly this command with --resume auto appended.
```

The initial real-model check correctly refused a native full-sequence replay with a large hidden-state mismatch. The fix replays the original target decoder calls, sparse tree masks, positions and KV operations, without sampling or draft generation and without loosening tolerance. The old replay gate on the real target then matched hidden states and full-softmax distributions exactly. A subsequent CLI check exposed tuple/list JSON identity in schedules; configuration serialization now uses JSON-native lists, with a regression test. Earlier logs are retained as development evidence and do not represent successful final runs.

The final real-model run, CLI resume, replot and report all exited successfully. The saved boundary has exactly 12 distinct responses (six conditions × two prompts), one metrics row, and no duplicate response keys after resume. Hidden-state maximum error and full-softmax TV are both **0**; the zero-drift Stale/Fresh loss/gradient/weights/optimizer check passed. OFF recorded zero updates and zero B norm; ON recorded three feedback updates per prompt and B norms approximately 0.03495/0.03926, equal under old/new target in this zero-drift control. All four A1/A2 effects and their smoke intervals are zero. Five plots were generated. Logs are `validation/qwen3b_final_{smoke,resume,replot}.log`, exported reporting is `validation/qwen3b_final_report.csv`, and machine-readable validation is [validation/result.json](validation/result.json).

For production prepare/validate, main training, resume, replot and CSV reporting, use the environment-variable commands in [README.md](README.md), pointing `DRAFT_CHECKPOINT` at the actual pretrained draft. Replot constructs no models and uses no GPU, but reads the durable checkpoint on CPU to recover committed boundaries.

## Exact metric columns

Each step has `policy_step`, `eval_prompts`, `eval_trajectories`, `evidence_label`, `research_result`, `training_trajectory`, `eval_max_new_tokens` and `aal_length_capped`.

For each prefix `stale_new`, `fresh_new`, `reflex_off_old`, `reflex_on_old`, `reflex_off_new`, `reflex_on_new`, columns are `<prefix>_aal`, `<prefix>_accepted_length_sum`, `<prefix>_verification_rounds`.

Effect columns are `A1_delta_lag`, `A2_gain_old`, `A2_gain_new`, `A2_drift_interaction`, each with `<effect>_ci_low` and `<effect>_ci_high`. Additional columns are `teacher_policy_tv_full_softmax`, `teacher_policy_tv_max`, `teacher_policy_drift_contexts`, `teacher_policy_drift_distribution`, `target_old_sha256`, `target_new_sha256`, `zero_drift`, `zero_drift_placebo_passed`, `old_replay_hidden_max_abs`, `old_replay_distribution_tv_max`, `shadow_stale_feature_loss`, `shadow_stale_distribution_loss`, `shadow_fresh_feature_loss`, `shadow_fresh_distribution_loss` and `boundary_wall_s`.

Per-response JSONL also records prompt/question IDs, accepted token IDs, seeds, raw verification counters, proposed/accepted proposal tokens, target/draft/A/head hashes, response cap/settings, OPD feedback updates/B norms, runtime and peak CUDA allocation. Each boundary saves its alignment gate, compact teacher trace, result journal and atomic completion marker. A checkpoint commits target LoRA/optimizer, R/A/optimizer, S/optimizer, pending gradients, sampler and all RNG states. B is transient and never checkpointed.

## Limits

The real smoke is a zero-update software control with one train prompt × two responses per learner and two held-out prompts × six conditions at a four-token response cap. Its rewards are uniform and no trajectories contribute a GRPO gradient; the required real optimizer call executes with LR=0. It does not establish model quality, nonzero 3B policy-lag effects, statistical power or full 8×8 throughput. Nonzero target drift and continuation after crash are exercised by the tiny CUDA integration fixture with explicit synthetic rewards.

The default 200-step, length-2048, 8×8 experiment was not run because the actual production pretrained checkpoint is unavailable. Its memory fit on the 24 GiB 3090 is unverified. Two draft optimizers, full-vocabulary teacher loss and long rollouts need substantially more memory than the reduced smoke. Old target/Fresh snapshots are owned CPU data; no second full target is constructed on GPU. The final smoke recorded a maximum per-response CUDA allocation of 11,508,200,960 bytes (10.72 GiB); this is an evaluation measurement, not a full-run memory guarantee. Its latest checkpoint is 3,985,981,466 bytes and teacher trace is 482,409,248 bytes. Atomic replacement temporarily needs another checkpoint-sized allocation on disk; CPU fork states require additional RAM. Multi-GPU training and reproducibility across different compiler/GPU versions are not claimed.

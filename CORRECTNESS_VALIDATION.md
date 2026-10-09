# Correctness validation — 2026-10-09

**B200 production execution is not verified.** This machine exposes only an RTX 3090 (24 GiB). The actual pretrained draft described by the task is on B200 and is absent from the supplied local `SpecNaacl` tree. No new draft was trained, initialized as a substitute, or downloaded.

## Corrections

| Verified defect | Correction and evidence |
|---|---|
| Unparseable gold previously awarded accuracy 1 | Validate gold before sampling; raise or record/exclude invalid rows. The official local train has two empty boxed labels (`5054`, `5158`): 8519 valid unique train questions remain; all 500 test questions remain. Nonempty exact-text parser fallback is recorded. |
| Uniform rewards and forced LR=0 steps were counted as policy transitions | Exclude zero-variance groups; skip target optimizer on empty/zero-gradient attempts, checkpoint sampler/RNG/learner updates, and retry. Count only actual target optimizer calls. Compare parameter hashes and measured TV independently. |
| Zero-drift control replaced a real transition and advanced Adam moments | Run an isolated, restored Stale/Fresh shadow placebo. Primary scheduled boundaries retain the genuine target update. |
| Long SDPA backward failed a bitwise-only control even with identical labels/RNG | A numerical diagnostic with the actual 3B draft/head and tiled saved features reproduced identical losses but gradient max error 5.96e-8. Placebo checks now compare actual gradients, weights, moments and losses within explicit dtype precision, recording every deviation. Tiled rows are a numerical diagnostic, not integration trajectories. Freeze/resume checks remain exact. |
| Short padded draft sequences predicted a fabricated zero feature/distribution at their final real position | Mask each sequence with `P <= i < L-1`, independent of packing. A packing-invariance test compares losses and gradients. |
| Empty masks divided by zero; last microbatch omitted accumulation scaling | Exclude empty examples from loss/normalization; apply the same valid-example and accumulation denominator to every microbatch. Empty-only batches do not advance Adam. |
| `log(softmax)` underflow; masked NaN/Inf; large FP32 vocabulary copies | Use stable log-softmax CE, mask before log-probability operations, stable `expm1` KL, chunked checkpointed CE/log-probabilities and target decoder gradient checkpointing. Nonfinite supervised loss/gradients stop execution. The objective remains mean-example(`2*SmoothL1 + 0.1*soft CE`) plus reference GRPO/KL. |
| Prefill EOS continued into verification | Terminate EOS rows immediately, preserve shifted histories, compact rows/KV/trace ownership and count zero real verification rounds. Both generators cover all-EOS, mixed-EOS and verified-EOS cases. |
| AAL zero denominators could fabricate counts or discard bootstrap cases | Log legitimate prefill-EOS 0/0 responses, require positive condition totals, report undefined bootstrap replicate counts and null CIs. All six conditions use the same ratio of sums. |
| Optional pretrained A rejected by draft loading | Strictly load the existing backbone into independent R/S. Load checkpoint A only into R; otherwise use the unchanged reference QR initializer and record `projector_pretrained=false`. Apply persistent analytical A gradients once per draft boundary. |
| Main OPD's completed KV pools overlapped shadow/replay cache allocations | Release completed target/draft KV capacity pools in the runner; the source `end_rollout(0)` retains them. A CUDA regression verifies unchanged tokens, features, counters, A feedback and RNG. |

Fresh still replays the exact saved shadow token IDs, positions, masks, tree attention and KV calls. Both teacher features and full-softmax CE labels change under the new target. A2 still freezes the same R/A for four conditions and resets B per rollout. OPD proposal, feedback, sampler, projector gradient, optimizer and KV kernel modules remain copies of the reference; the numerical/mask/EOS exceptions above are explicit.

## Executed checks

| Check | Result |
|---|---|
| CUDA suite, Torch 2.6.0+cu124 / Transformers 5.12.1 / PEFT 0.19.1 / Triton 3.2.0 | **340 passed**, 85 warnings, 23.52 s. Includes real CUDA/Triton tiny-model GRPO, replay, A2, EOS, isolated control, crash/resume, skipped-attempt recovery, idle-KV release and evidence validation. Tiny-model rewards are synthetic only in the explicitly marked fixture. |
| CPU suite, Torch 2.13 CPU / Transformers 5.12.1 | **100 passed, 213 skipped**, one intentional invalid-label warning, 4.37 s. CUDA tests skip on this environment. Includes evidence-checker rejection of missing resume, duplicate records and absent ON feedback under the new target. |
| Real SimpleLR preparation | PASS: official disjoint train/test, duplicate and gold audit, deterministic seed-2026 fixed 64 test IDs. |
| Local port hashes | PASS: 57 mapped files verified. |
| Shell syntax and configurable launch | PASS: main/B200 scripts parse; dry-run forwards absolute model/draft/data paths. |
| Actual B200 preflight | Correctly rejects `NVIDIA GeForce RTX 3090, capability (8,6)`. |

Logs are under [validation/correctness_fixes/](validation/correctness_fixes/). Earlier `focused_initial`, `focused_v2` and `cuda_tests_v3` logs are development history; `cpu_tests_final.log` and `cuda_tests_final.log` are the completed suites.

## Real-model integration and memory

The local integration uses the actual existing `/mnt/hdd/nhatminh/SpecDecode/models/Qwen2.5-3B-Instruct`, official local SimpleLR Parquet files and the only discovered compatible existing checkpoint: `/mnt/hdd/nhatminh/.specnaacl-validation/qwen3b-modern/latest_checkpoint`. That checkpoint is a **2-step pretrain validation artifact**, SHA256 `ddadc931fcafe9caeb45ddcc428d8e0912d6aa59a62fe8bc0cfe7c6f8830b278`; it is not the production pretrained draft and is never a default. Integration uses natural math/format rewards; no synthetic rewards or forced target LR are injected.

The two initial full-shape attempts kept 8×8 per learner, total sequence length 2048 and scheduled 16-prompt evaluation. They failed before any completed measurement boundary:

- Default allocator: requested 892 MiB, PyTorch reported 20.82 GiB allocated + 1.85 GiB reserved unused, with 535.94 MiB device memory free.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: the main OPD rollout completed; the shadow then exhausted memory in the reference full-vocabulary top-p sort (`helper/fastgrpo_generate.py:92`). It requested 808 MiB with 775.94 MiB device memory free, 22.28 GiB allocated and only 163.07 MiB reserved unused. Sampled whole-device usage reached 24105 MiB. Subsequent investigation found that idle main KV pools were still retained; the final runner releases them before shadow/replay allocations.

These are failed validations, not training results. A separate reduced pilot used the same real target/data/checkpoint and natural rewards, with 2 prompts × 8 responses, total sequence length 2048, two evaluation prompts and a 16-token evaluation cap. It reached gradient preparation and passed old-teacher replay, then exposed the bitwise-only control failure corrected above. It completed no target optimizer step or measurement boundary. Final execution evidence is in `validation/correctness_fixes/result.json`.

**The final local full-shape run passed one actual boundary** after the KV-pool and numerical-control fixes. Output: [validation/correctness_fixes/qwen3b_8x8_cache_release/](validation/correctness_fixes/qwen3b_8x8_cache_release/). It preserves 8×8 per learner, length 2048, N=16 and evaluation cap 256; `validation_run=true`, `research_result=false`. Main generated 64 responses / 28895 tokens and shadow 64 / 30340. All eight natural reward groups had variance. Both draft optimizers and the target optimizer stepped once; target gradient norm was **0.06630265**, A gradient norm **0.87309849**. Target parameter hashes changed and prompt-final policy TV was **4.624655e-5** (maximum **3.425705e-4**). Old replay hidden/distribution errors were exactly zero across all 64 saved shadow histories. A changed from its explicit new QR initialization and then remained identical across all four A2 conditions.

| Condition | Responses | Accepted length / rounds | AAL |
|---|---:|---:|---:|
| Stale/new | 16 | 3714 / 2823 | 1.315622 |
| Fresh/new | 16 | 3714 / 2810 | 1.321708 |
| Reflex OFF/old | 16 | 3823 / 2869 | 1.332520 |
| Reflex ON/old | 16 | 3757 / 2813 | 1.335585 |
| Reflex OFF/new | 16 | 3681 / 2808 | 1.310897 |
| Reflex ON/new | 16 | 3668 / 2705 | 1.356007 |

OFF had zero updates and zero B norm. ON recorded 2813/2705 feedback updates under old/new target, with positive final B norms. The isolated zero-drift control passed: identical losses, max gradient error 1.676381e-8, weight/moment errors within declared dtype tolerance; persistent training state was restored. These are execution checks with a validation draft, not research evidence about a pretrained production method.

Peak PyTorch training allocation was **20,017,797,120 bytes (18.643 GiB)**; peak evaluation allocation **10.891 GiB**; sampled whole-device peak **21045 MiB**. The boundary took 2244.72 s before journal/checkpoint writes, with 2310.39 s reported including commit. The checkpoint is 3,985,982,810 bytes and teacher trace 869,280,145 bytes. Five plots were rebuilt from the committed journal. This verifies this single boundary's memory fit on the local 3090; longer trajectories and the production draft remain unmeasured.

Actual checkpoint resume **passed exact state identity** for target LoRA, all three optimizer states, R/S weights and gradients, pending A sums/weight, sampler and all RNG state. No duplicate response records were produced. `scripts/verify_execution.py` reports **PASS**, one real update, one boundary and 96 evaluation responses, while explicitly recording `b200_execution_verified=false`. The current implementation hashes also match the completed run's manifest. The production B200 completion criterion remains unmet.

## Blockers and launch

Required remaining evidence is an actual B200 run with the actual production draft, natural nonzero target gradients/parameter changes, a completed six-condition boundary and exact resume audit. Neither B200 nor a remote host/access path was provided. The local Torch 2.6/CUDA 12.4 build lacks `sm_100`; a B200 CUDA >=12.8 build is required. The pinned Torch 2.8 CUDA 12.8 wheel is listed by the [official PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/). B200 preflight checks the installed device/build before allocating models.

```bash
cd /path/to/analysis_spec
export PYTHON_BIN=/path/to/b200-cuda-env/bin/python
export MODEL=/absolute/path/to/Qwen2.5-3B-Instruct
export DRAFT_CHECKPOINT=/absolute/path/to/SpecNaacl/existing-pretrain/latest_checkpoint
export TRAIN_DATASET_PATH=/absolute/path/to/simplelr_abel_level3to5/train.parquet
export TEST_DATASET_PATH=/absolute/path/to/simplelr_abel_level3to5/test.parquet

# One actual 8x8 boundary, N=16, checkpoint/resume and evidence verification.
OUTPUT_DIR="$PWD/outputs/b200_validation" bash scripts/validate_b200.sh

# Main 200-step experiment, unchanged required schedules and N=16/64.
OUTPUT_DIR="$PWD/outputs/qwen3b_a1_a2" bash run_policy_lag_motivation.sh --require-b200
# Resume with exactly the same config, paths and OUTPUT_DIR; add --resume auto.
```

Only `analysis_spec` was edited by this work. The read-only reference audit preserves its before snapshot: 427 of 428 reference files match; `.gitignore` changed concurrently outside these edits. All reference runtime source files match the task's snapshot. The earlier initial-port audit separately recorded external driver/helper changes and remains preserved.

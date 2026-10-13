# RQ2: AAL throughout a rollout

This is a separate frozen-checkpoint evaluation. It does not train or resume
training and does not overwrite E2/E3/E6 or repeated-miss RQ2 data. The new
runner explicitly selects prompt-group SPARK; existing generator calls retain
their default batch-wide behavior.

## Required checkpoint and provenance

The default and intended checkpoint is the single read-only SpecNaacl resume
checkpoint:

```text
/workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/train/qwen25_3b/qwen25_3b__simplelr__method-opd_reflex__seed42__20261008T104039__3885bd48/checkpoints/resume/step234_epoch2_batch0.pt
```

Preflight uses the existing `rq2.checkpoints.load/decode` path, requires native
`opd_fastgrpo_checkpoint_v5`, method `opd_reflex`, matching step (234), finite
weights and learned-A provenance. It strictly restores the draft and target
LoRA on CPU with a meta base model, validates learned A's shape/dtype, and checks
their hashes before any GPU model is allocated. GPU restoration is also checked.
No standalone target/draft exports, projector initialization, optimizer restore,
or pretrained-draft fallback occurs.

The manifest records checkpoint SHA256, target/draft hashes, learned A hash,
training metadata, sibling resolved-config hash, dataset/model/tokenizer hashes,
decoding settings, prompt IDs/hashes, seeds, implementation hashes, and package
versions. Absent checkpoint B-sharing metadata is interpreted as SpecNaacl's
legacy `rollout_batch` default. A conflicting later sibling config does not
override checkpoint provenance. Such a checkpoint is explicitly described as
**per-prompt adaptation evaluation on a checkpoint trained with batch-wide B**;
it is not labeled fully per-prompt-trained SPARK.

## Per-prompt adaptation

`rq2/per_prompt.py` specializes the existing OPD engine.
`rq2/per_prompt_kernels.py` ports the group-sensitive kernels from read-only
SpecNaacl. Teacher extraction, proposal normalization, the union/tail objective,
target sampling, tree verification, KV/history optimizations and passive event
logging reuse existing analysis_spec utilities.

There is one full-vocabulary `B[g, token, rank]` per original prompt. All eight
responses of that prompt share it. Feedback's weighted numerator and denominator
are both restricted to that prompt:

```text
B[g] -= fast_lr * sum_{selected states of g, union token v} w_s (q_s(v)-p_s(v)) u_s
                 / sum_{selected states of g} w_s
```

A prompt with no selected feedback receives no update. Original response IDs
map to prompt IDs by integer division by responses-per-prompt. Physical owners
are updated after prefill EOS removal and KV swap-removal; canonical sampler
rows map back to physical cache rows for feedback. Ownership arrays used by
asynchronous feedback retain their lifetime until its stream finishes.

A is loaded directly from the checkpoint and frozen. B resets before every
arm rollout and is cleared afterward. OFF uses zero learning rate and leaves B
identically zero; ON updates B after each verification round. Both arms share
the prompt batch, seed, target/draft/A, sampling configuration and proposal
budget. No extra target forward is added. Production validation checks target
forward count, weight hashes, group feedback/update counters, finite B, zero B
before/after rollout, and log/counter reconstruction. Smoke additionally replays
without passive logging and compares tokens, acceptance counters and RNG.

Batch-wide proposal timing profiles are not reused as calibrated per-prompt
profiles. Dispatch retains the existing uncalibrated sparse/fused/GEMM paths.

## AAL and statistics

Each OFF/ON rollout normalizes its own verification-round sequence to progress
using round midpoints `100*(round_index+0.5)/total_rounds`. Integer arithmetic
assigns rounds to `[0,20), [20,40), [40,60), [60,80), [80,100]` percent bins.
Very short rollouts can have empty bins; prefill-only EOS has no verification
measurements. Missing bins remain missing and are never imputed.

Within each bin:

```text
AAL = sum(accepted_length) / sum(active_response_rounds)
```

The numerator includes the terminal verification token on the accepted path,
matching native `total_acc_length`. The denominator matches native
`total_decoded_token_num`, the sum of per-response verification-round counts.
Prefill is excluded. This is not a mean of per-round AAL and is not the ratio
of accepted speculative tokens to proposed speculative tokens.

For each progress bin, OFF and ON are summarized over the same paired valid
rollouts, with equal weight per rollout. A 2,000-resample bootstrap resamples
whole rollout pairs, preserving OFF/ON pairing and missing-bin patterns.
One supported rollout gives a point estimate without a fabricated confidence
interval. CSV records paired/individual valid counts, missing rollout IDs,
verification rounds, active response-rounds, mean/min/max active responses,
accepted lengths, AAL, and paired ON-minus-OFF differences. No increasing-AAL
trend is assumed or enforced.

Nonfinite probabilities, correction features, B or AAL are rejected. As in
existing RQ2 numerical validation, a probability-space KL diagnostic may be
undefined from FP32 underflow/rounded-zero tail despite a finite `q-p` update;
its counter is retained and its overall mean is null, never fabricated.

## B200 commands, GPU 3

Activate the existing model environment first. Copy-paste:

```bash
cd /workspace/storage-shared/nlp/minhpn19/analysis_spec
export CUDA_VISIBLE_DEVICES=3
export PYTHON_BIN=python3
export MODEL=/workspace/storage-shared/models/Qwen2.5-3B-Instruct
export TEST_DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet

AAL_ARGS=(
  --checkpoint-path /workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/train/qwen25_3b/qwen25_3b__simplelr__method-opd_reflex__seed42__20261008T104039__3885bd48/checkpoints/resume/step234_epoch2_batch0.pt
  --checkpoint-step 234
  --output-dir "$PWD/outputs/rq2_aal_step234"
  --num-rollouts 20 --prompts-per-rollout 8 --responses-per-prompt 8
  --seed 2026 --temperature 1.0 --top-p 0.95 --max-length 2048
  --verification-capacity 512 --max-verification-num 160
  --max-draft-k 8 --max-draft-token-length 5 --min-draft-token-length 3
  --opd-topk 16 --opd-fast-lr 0.01 --opd-update-stream
  --bootstrap-samples 2000 --bootstrap-seed 2026 --wall-minutes 150
)

# CPU checkpoint/model/data preflight; no generation or training.
bash run_rq2_aal_progress_b200.sh check "${AAL_ARGS[@]}"

# One rollout pair (8 prompts x 8 responses), plus passive-logging replays.
bash run_rq2_aal_progress_b200.sh smoke "${AAL_ARGS[@]}"

# Validate/reuse smoke, collect 20 rollout pairs, then analyze.
bash run_rq2_aal_progress_b200.sh all "${AAL_ARGS[@]}"
```

`all` runs smoke automatically, so the explicit `smoke` command is optional.
Keep `--num-rollouts 20` in smoke configuration: smoke mode already selects only
the first rollout. This keeps its manifest compatible with the full run.

To split full collection and offline analysis after smoke:

```bash
bash run_rq2_aal_progress_b200.sh collect "${AAL_ARGS[@]}"
CUDA_VISIBLE_DEVICES='' python3 run_rq2_aal_progress.py analyze \
  --output-dir "$PWD/outputs/rq2_aal_step234" \
  --bootstrap-samples 2000 --bootstrap-seed 2026
```

The 150-minute budget is checked between arm rollouts, not inside a model
forward. Completed arms commit independently and safely; unpaired arms are
excluded from analysis. Repeat the same command to resume evaluation commits.
There is no training resume. A different checkpoint/config/code identity
requires a different new `outputs/rq2_aal_*` directory. Existing RQ2 manifests
also retain their original strict source-hash checks.

## Outputs

Everything is under `analysis_spec/outputs/rq2_aal_step234/`:

```text
manifest.json                 immutable checkpoint/config/input identity
preflight.json                CPU restore and training provenance
runtime.json                  actual CUDA hardware/build
smoke_execution.json          smoke runtime/status
smoke/{off,on}/rollout_0000/   smoke commits/events/rounds/tokens/diagnostics
smoke_analysis/               smoke-only figures/statistics, never production
collection/{off,on}/rollout_*/ committed full-evaluation logs
collection_execution.json     collection runtime/status
aal_by_rollout.csv            5 bins per arm and rollout
aal_summary.csv               means/CIs/paired differences and coverage
aal_during_rollout.pdf
aal_during_rollout.png
active_responses.pdf
validation.json               counts, provenance, counters, timings, hashes
```

No results are generated if there are no fully committed OFF/ON pairs. The
implementation's local correctness tests and production preflight availability
are recorded separately under `validation/`; synthetic fixtures do not confirm
the scientific hypothesis or B200/3B performance.

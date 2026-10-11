# Pure FastGRPO E2–E6 motivation pipeline

This is a new standalone pipeline inside `analysis_spec`. `SpecNaacl` is a
read-only reference and supplies the existing pretrained draft as input data.
No runtime code is imported from `SpecNaacl`. No SPARK, Reflex projector, B_fast,
or Fresh/Stale branch is constructed. E2–E6 all consume the same visited-path
verification event log per checkpoint; analysis runs without any model forwards.

## 1. B200 environment and paths

Run these exports in the activated B200 Python environment. The deployment below
places `analysis_spec` beside `SpecNaacl` under the shared project directory.

```bash
cd /workspace/storage-shared/nlp/minhpn19/analysis_spec

export PYTHON_BIN=python3
export CUDA_VISIBLE_DEVICES=0
export MODEL=/workspace/storage-shared/models/Qwen2.5-3B-Instruct
export TRAIN_DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet
export TEST_DATASET_PATH=/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet
export DRAFT_CHECKPOINT=/workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint
export OUTPUT_DIR="$PWD/outputs/e2_e6_fastgrpo"
export SEED=42

# Verify the reference inputs exist before starting.
test -d "$MODEL"
test -f "$TRAIN_DATASET_PATH"
test -f "$TEST_DATASET_PATH"
test -f "$DRAFT_CHECKPOINT/draft.pth"
test -f "$DRAFT_CHECKPOINT/target_config.json"
"$PYTHON_BIN" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

`--require-b200` performs the existing B200 device/build preflight. The launcher
keeps caches, checkpoints, logs and all outputs under `analysis_spec` and refuses
an output directory outside that tree. It sets offline loading and pure FastGRPO.
There is no fallback to the local 3090 validation draft or an untrained draft.
If the reference draft's provenance config is stored elsewhere, export
`DRAFT_TARGET_CONFIG=/absolute/path/to/target_config.json` and pass/use it through
the launcher. Checkpoint architecture, target config, vocabulary, special tokens,
model/data hashes, rendered prompts and train/test overlap are validated.
The training manifest also pins runtime source hashes and installed package versions.

## 2. Train to 500 completed target optimizer steps

```bash
bash run_e2_e6.sh train \
  --require-b200 \
  --train-steps 500 \
  --batch-size 8 --responses-per-prompt 8 \
  --accumulation-steps 4 --draft-accumulation-steps 1 \
  --target-lr 1e-6 --draft-lr 1e-4 \
  --beta 0.04 --epsilon 0.1 \
  --temperature 1.0 --top-p 0.95 \
  --verification-capacity 512 --max-verification-num 160 \
  --max-draft-k 8 --max-draft-token-length 5 \
  --min-draft-token-length 3 --draft-token-length-c 0.75 \
  --max-length 2048 --max-prompt-length 2048 \
  --max-training-token 1024 --max-training-padding-gap 4096 \
  --save-every 10
```

The local `helper/fastgrpo_generate.py`, original online feature/soft-CE draft
objective and target GRPO loss are reused. One persistent draft and target are
initialized from the reference Qwen2.5-3B configuration, pretrained draft, and
rank-64/alpha-32 target LoRA. Rewards are accuracy + 0.2 × format reward, with
per-prompt group-standardized advantages. Target examples follow the reference's
decode/retemplate/tokenize path; draft supervision uses the actual rollout history.
Prompt templates and official Parquet splits use the existing local utilities.
Invalid gold labels excluded by those utilities are recorded in the dataset manifest.

**Step semantics:** inspection of `SpecNaacl/grpo_speculative.py` shows that the
source calls the target optimizer after each rollout having nonempty eligible
reward groups. Its `accumulation_steps=4` is used in the source's `used_items`
reporting divisor, not to defer those optimizer calls. This pipeline preserves
that actual update cadence and counts completed target `optimizer.step()` calls
directly. A zero-variance group contributes no target examples. If every group
is excluded, or the target gradient is zero, the target step is skipped and the
policy counter does not advance. The online draft can still learn from that
rollout, exactly as the reference trains it before testing target eligibility.

Milestones at 30/100/200/500 are permanent. Resume checkpoints are committed every
10 completed target steps, at the final requested step, and on graceful stop.
SIGINT/SIGTERM set a stop flag; the current rollout/update finishes and a complete
state is saved at the next safe boundary. A second command resumes from the last
durable commit. An abrupt process/machine crash may replay work after that durable
commit, but never replays the committed optimizer steps.

## 3. Resume after interruption

Use the same exported paths and original training settings:

```bash
bash run_e2_e6.sh resume --require-b200
```

Target LoRA, pure draft, both AdamW states, RNG states, shuffled sampler/cursor,
policy/attempt/draft-update counters and cumulative elapsed time are restored.
There are no pending gradient accumulations at a save boundary. The exact resume
audit is written to `$OUTPUT_DIR/resume_audit.json`. Diagnostics newer than the
last durable attempt are removed from the resumable training log.

The authoritative commit pointer is `checkpoints/latest.json`; `latest.pt` is a
convenience hardlink. Integrity sidecars contain SHA256, byte count and weight
hashes. Milestones are hardlinked from committed generations, stay immutable,
and are never deleted by cleanup. The latest two resume generations are retained.
A crash between latest commit and milestone publication is repaired on resume.

## 4. Collect checkpoints 30/100/200 only

Training can stop before 500; any completed milestones remain usable:

```bash
bash run_e2_e6.sh collect \
  --require-b200 \
  --checkpoints 30,100,200 \
  --eval-rollouts 20 --eval-seed 2026
```

Each rollout has 8 held-out prompts × 8 responses. The official test set is
shuffled deterministically; the first 160 **distinct** prompts form 20 disjoint
rollouts. The same prompt plan is reused at each checkpoint. There are no repeated
prompts across rollouts, which prevents cross-rollout control leakage. Requests
requiring more distinct prompts than the test file contains fail explicitly.

For each rollout, collection runs:

1. The actual pure FastGRPO generator with a compact visited-path event hook.
2. Exact response reconstruction and path/classification checks.
3. A separate event-logging-disabled replay with identical checkpoint, prompts,
   seed and decoding settings; all generated tokens and acceptance counters must match.
4. Atomic Parquet/JSON writes, file checksums, then a completion commit marker.

No additional target/draft forwards are inserted inside a logged generation.
The timing replay is a separate generation required for performance measurement.
The original multi-response `max_length=2048`/EOS stopping is preserved; a new
single-response token cap is not imposed on the 8×8 rollouts.

Event logs contain prefill plus every **visited** verification-path token,
canonical response/group/prompt IDs, round and position, target sample, draft
top-1 and the full top-k candidate IDs **before pruning**, retained child IDs,
observed draft k/budget/actual nodes including the verification root, accepted
length and preceding contexts of 2/4/8 tokens and full-prefix hash.
Unvisited branches are never labelled rejected. A non-expanded beam frontier
has no computed draft distribution and is explicitly `beam_cap`; final-depth
leaves are `depth_cap`. EOS, beam/depth/length caps and prefill are censored from
ordinary miss-rate denominators. No distribution is fabricated at these contexts.

Every event in verification round r becomes observable at the **end of r**.
Events from the same batched verification computation cannot supply causal
feedback to each other. All feedback analyses require strictly smaller rounds.
Swap-removal uses the generator's canonical `residual_index` mapping.

## 5. Include step 500 later, without repeating earlier valid rollouts

```bash
bash run_e2_e6.sh collect \
  --require-b200 \
  --checkpoints 30,100,200,500 \
  --eval-rollouts 20 --eval-seed 2026
```

Completed valid rollouts are verified and skipped; their files are not rewritten.
Missing checkpoints are reported and skipped. A changed collection identity or a
corrupt finalized rollout fails instead of silently mixing experiments. A rollout
interrupted before its completion marker is the only work repeated on resume.

For a short check using genuine available milestone checkpoints:

```bash
bash run_e2_e6.sh collect --require-b200 --checkpoints 30,100,200 --smoke
```

`--smoke` collects the first 8×8 rollout of the same immutable 20-rollout plan
per available checkpoint. A later ordinary `collect` command skips that valid
rollout and fills the remaining 19. It still requires genuine
checkpoints and the production draft; it does not relabel validation checkpoints
as training milestones. Keep `--eval-rollouts` unchanged between these commands;
the smoke flag limits execution, without changing the planned dataset.

## 6. Offline analysis and PDF figures

```bash
bash run_e2_e6.sh analyze \
  --checkpoints 30,100,200,500 \
  --bootstrap-samples 2000 --eval-seed 2026
```

This command requires no GPU, base model, or pretrained draft access. It validates
committed logs and resumes from per-checkpoint analysis caches. Adding checkpoint
500 does not recompute valid unchanged analyses of 30/100/200. Partial collection
results are supported and their actual rollout counts are reported.

- **E2:** baseline miss probabilities conditioned on observed target token y;
  within-response, within-rollout and cross-rollout recurrence and confusion-pair
  risks; round distances 1, 2–4, 5–16, 17–64 and >64. Cross-rollout controls reserve
  disjoint donor/recipient halves, one donor per recipient, and inspect donor
  history strictly before the recipient round. Controls never reuse recipient
  feedback; their independent donor/recipient pairs remain within rollout clusters.
- **E3:** mutually exclusive same-response, sibling-response, different-prompt
  sources, with exact preceding 2/4/8-token and full-context matches.
- **E4:** host replay wall shares, miss rate, AAL, recurrence share, observed
  proposal budget, actual nodes/response and coverage/pruning shares for active
  sizes 64, 32–63, 16–31, 8–15 and <8.
- **E5:** coverage/pruning miss proportions by checkpoint, actual context depth,
  active batch size and draft k. No unmeasured counterfactual budget is inferred.
- **E6:** already available token/confusion-pair/context feedback among miss events,
  separately for response, inclusive group and inclusive rollout scopes. These
  include overall, round-position, response-position and prior-token-exposure
  strata. They are availability measurements, not claims of successful adaptation.

Matched lifts use actual target-token ID, round/position bins, context depth,
observed draft k, active-size bin and preceding-exposure bins as strata. Exposure
counts include all earlier hits and misses, with separate opportunity and actual
target-token counts in the relevant response/group/batch scope; confusion pairs
also condition on draft top-1 and prior pair occurrences. Cross controls include
donor opportunity, target-token and pair exposure. Both exposed/unexposed risks are
recomputed in 2,000 whole-rollout bootstrap draws on common support, with fixed
empirical stratum weights. Raw descriptive rates and all denominators are also
reported. A missing common-support stratum or undefined estimate is null, not zero.
Fewer than two contributing rollout clusters yields no confidence interval.
Checkpoints are never pooled. Groups/responses stay inside rollout clusters;
collection uses distinct prompts across clusters. Cross-control availability is
computed from the fixed independent donors and bootstrapped over recipient rollouts.

**Timing limits:** total replay time is measured with CUDA synchronization only
at rollout boundaries. No new per-round synchronization is inserted. Active-bin
shares are measured CPU phase-wall intervals; asynchronous GPU carry-over can
cross phase boundaries. The remainder, prefill and postprocessing are not hidden
inside these bins. CUDA stream-event elapsed intervals are reported separately;
they include stream idle/host scheduling gaps and are **not kernel-only times**.
Event-logging runtime is always separate and is never presented as baseline speed.

Outputs under `$OUTPUT_DIR`:

```text
train_manifest.json                 model/draft/data/tokenizer/config/source identities
runtime_preflight.json               GPU/build validation
checkpoints/step_000030.pt           immutable milestones (+ integrity JSON)
checkpoints/step_000100.pt
checkpoints/step_000200.pt
checkpoints/step_000500.pt
checkpoints/latest.json             authoritative resume commit
checkpoints/latest.pt               convenient latest hardlink
logs/train.jsonl                    rewards/losses/AAL/acceptance/tokens/time/memory/steps
logs/collect.jsonl                  seeds/counts/replay/logging time/checkpoint diagnostics
logs/failures.jsonl                 errors and tracebacks
collection/manifest.json            immutable 20-rollout prompt/seed/generation plan
collection/step_*/rollout_*/         events.parquet, rounds.parquet, responses.json,
                                    timing.json, complete.json with file hashes
analysis_e2_e6/E2.csv ... E6.csv      tables with counts and 95% CIs
analysis_e2_e6/E2.json ... E6.json    JSON tables
analysis_e2_e6/summary.json          separate checkpoint summaries
analysis_e2_e6/recurrence.pdf
analysis_e2_e6/response_sharing.pdf
analysis_e2_e6/long_tail.pdf
analysis_e2_e6/miss_types.pdf
analysis_e2_e6/REPORT.md             measured positive/negative/inconclusive associations
analysis_e2_e6/manifest.json         exact analysis input hashes/bootstrap/source identity
```

An optional sequential orchestration command is available; it is never launched
automatically during implementation:

```bash
bash run_e2_e6.sh all --require-b200 --train-steps 500 --checkpoints 30,100,200,500
```

It resumes existing training when appropriate, reuses one model in-process for
collection, then performs offline analysis. Independent commands are preferable
when training and collection run at different times.

## 7. Logs and integrity checks

```bash
tail -f "$OUTPUT_DIR/logs/train.jsonl"
# In another terminal:
tail -f "$OUTPUT_DIR/logs/collect.jsonl"
cat "$OUTPUT_DIR/training_status.json"
cat "$OUTPUT_DIR/resume_audit.json"

# Verify available milestone/latest hashes and every committed rollout log.
bash run_e2_e6.sh check --checkpoints 30,100,200,500

cat "$OUTPUT_DIR/analysis_e2_e6/REPORT.md"
```

## Executed validation and unverified behavior

The current development host has an RTX 3090, not B200. CPU tests cover response
reconstruction, canonical swap-removal, visited-path labels, coverage/pruning,
strict causality, exclusive source categories, EOS/censoring, response-context
matching, cluster bootstrap, common support, immutable checkpoint selection,
corruption and crash recovery. CUDA tests cover token/counter/RNG invariance,
event-disabled timing replay, exact interrupted-vs-uninterrupted pure training,
zero-variance skipping and collection/analysis/resume through tiny checkpoint
fixtures. Fixture step IDs are **not** actual 30/100/200/500 training milestones.

Executed checks: full CUDA-environment suite **375 passed in 59.67 s**; final
E2–E6 checks **22 passed in 26.92 s**; CPU-only environment **15 passed, 7 CUDA
tests skipped in 5.68 s**. The tiny fixture's four one-rollout checkpoints used
0.509 s for event logging plus 0.501 s for equivalent replay (generation time
only, excluding model loading, checkpoint I/O and offline analysis).

The B200 production paths and pretrained draft are unavailable on this host.
No validation-only draft is substituted. No 500-step training or production E2–E6
measurement is launched or claimed. See `validation/e2_e6_validation.json` for
actual test results, runtime, reference-source audit and limitations.

To run tests locally, keep their scratch files inside `analysis_spec`:

```bash
PYTHONPATH="$PWD/.testdeps:$PWD" "$PYTHON_BIN" -m pytest -q \
  --basetemp "$PWD/.cache/e2_e6_pytest"
```

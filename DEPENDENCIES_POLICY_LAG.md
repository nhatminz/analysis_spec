# Pinned dependency contract

- FastGRPO: `yedaotian9/FastGRPO@38e252493149072d2c5905f0a47de1d935d7170a`
- SpecForge: `sgl-project/SpecForge@3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83` (`0.2.0`)
- Python: `>=3.11` (required by the pinned SpecForge commit)
- PyTorch: upstream pins `2.13.0`; the bundled source also contains its explicit
  Torch 2.11 CuteDSL compatibility shim, and launch validation accepts
  `2.11.x` or `2.13.x` without replacing the installed CUDA build.
- Transformers: upstream pins `5.12.1`; capability validation accepts
  `>=5.8,<6.0` for the offline B200 stack, including the installed `5.8.1`.
- SGLang: upstream SpecForge pins `0.5.18`; the offline-capture adapter also
  accepts the installed B200 `0.5.14` exactly. The adapter handles the missing
  runtime flags/DCP fields and the older `ModelRunner`, `ForwardBatch`, and
  DP-sync/request-range representations. SGLang is **not** used for FastGRPO
  rollout or verification here.

SpecForge source is bundled in `third_party/SpecForge`, including a
`VENDORED_COMMIT` provenance file, so the experiment does not clone or fetch
source code on the B200 machine. `requirements-policy-lag.txt` documents the
Python package contract without a Git URL. CUDA framework binaries are not
portable project source and must already exist in the B200 environment or be
provided through a local wheelhouse. The analysis
code imports SpecForge EAGLE-3's `AutoDraftModel`, `OnlineEagle3Model`, feature
layer rule, model forward, compact full-vocabulary teacher projection, and
training-time TTT objective. It never substitutes the local legacy
SmoothL1/soft-target draft objective for an EAGLE-3 branch.

The development workstation does not have this exact production environment,
so full B200 execution was not launched there. Dependency validation
intentionally fails on a mismatched stack instead of silently selecting another
SpecForge commit or objective.

The launcher imports the concrete EAGLE-3, FlexAttention, and offline SGLang
capture APIs before allocating the target model. It writes the actual Python,
Torch, Transformers, and SGLang versions plus local compatibility patches to
each pretrain run's `dependencies.json`; this records deviations from the
upstream lock instead of pretending the installed versions match it.

For targets with `tie_word_embeddings=true`, the frozen SpecForge target head
uses the checkpoint's configured embedding tensor when `lm_head.weight` is
deduplicated from the weight index. This is weight tying, not random or
reinitialized target supervision.

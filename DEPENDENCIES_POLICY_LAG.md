# Pinned dependency contract

- FastGRPO: `yedaotian9/FastGRPO@38e252493149072d2c5905f0a47de1d935d7170a`
- SpecForge: `sgl-project/SpecForge@3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83` (`0.2.0`)
- Python: `>=3.11` (required by the pinned SpecForge commit)
- PyTorch: `2.13.0` (SpecForge pin)
- Transformers: `5.12.1` (SpecForge pin)
- SGLang: `0.5.18` (SpecForge dependency, but **not** used for FastGRPO rollout or verification here)

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

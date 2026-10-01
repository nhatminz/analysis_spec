# Vendored SpecForge source

This is the runtime subset of `sgl-project/SpecForge` at commit
`3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83` (version `0.2.0`). It contains the
complete `specforge` Python package plus the upstream Qwen2.5-7B EAGLE-3
config, offline colocated recipe, and data/hidden-state preparation entrypoints
used by this experiment. The Qwen2.5-3B config beside it was deterministically
derived from the local target `config.json` using SpecForge's target-derived
EAGLE-3 field rules; it fixes the capture layers to `[1, 17, 32]` and uses the
same 16K draft vocabulary setting as the upstream Qwen2.5 recipes. Tests,
website documentation, CI files, and unrelated example assets were omitted;
no runtime algorithm files were modified.

Production launchers use `scripts/generate_eagle3_config.py` to apply those
same rules to the selected `TARGET_MODEL_PATH`; the checked-in 3B/7B files are
reference configurations, not a hard-coded model-size switch.

The upstream license is preserved in `LICENSE`. `VENDORED_COMMIT` is read by
the FastGRPO dependency validator so an offline copy does not need `.git`
metadata.

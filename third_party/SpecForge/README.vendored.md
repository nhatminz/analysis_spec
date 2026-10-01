# Vendored SpecForge source

This is the runtime subset of `sgl-project/SpecForge` at commit
`3cb0510f0bd0e8c195ac6e9c5c62f6b50580ff83` (version `0.2.0`). It contains the
complete `specforge` Python package plus the Qwen2.5-7B EAGLE-3 config, offline
colocated recipe, and data/hidden-state preparation entrypoints used by this
experiment. Tests, website documentation, CI files, and unrelated example
assets were omitted; no runtime algorithm files were modified.

The upstream license is preserved in `LICENSE`. `VENDORED_COMMIT` is read by
the FastGRPO dependency validator so an offline copy does not need `.git`
metadata.

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
the EAGLE-3 architecture, feature extraction, loss, and unrolling code is
unchanged.

Local runtime compatibility patches are applied in
`specforge/offline_capture/sglang_backend`: the optional
`sglang.srt.runtime_context.get_flags` API is imported lazily, and the
`ParallelState` constructor omits DCP fields on SGLang 0.5.14 where those
fields do not exist. The offline runner also translates the 0.5.18
`ModelRunner`/`ForwardBatch` call signatures and request `extend_range` into
their 0.5.14 equivalents. These adaptations preserve the rank-0/size-1
topology of the default DP/DCP-disabled capture path; requesting unsupported
DP/DCP modes still fails explicitly. They do not change captured tensors or
the training algorithm.

`TargetHead` also follows the target config's `tie_word_embeddings` contract:
when a tied checkpoint omits a duplicate `lm_head.weight`, it loads the shared
`model.embed_tokens.weight` tensor. Untied checkpoints still require an
explicit LM-head tensor and fail if it is missing.

Production launchers use `scripts/generate_eagle3_config.py` to apply those
same rules to the selected `TARGET_MODEL_PATH`; the checked-in 3B/7B files are
reference configurations, not a hard-coded model-size switch.

The upstream license is preserved in `LICENSE`. `VENDORED_COMMIT` is read by
the FastGRPO dependency validator so an offline copy does not need `.git`
metadata.

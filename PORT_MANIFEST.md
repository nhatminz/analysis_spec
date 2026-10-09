# Port and retirement manifest

Read-only authority: the actual `SpecNaacl` checkout supplied in this workspace, audited 2026-10-09. No reference file was edited by this implementation. The standalone experiment never imports or adds that tree to PYTHONPATH. It may read a user-supplied pretrained checkpoint as input data. Copying the checkpoint and companion target config elsewhere removes even that input-location association.

Port hashes identify the original audited snapshot. A final audit observed concurrent external edits to the reference driver and two new initialization/alignment helpers; all copied runtime kernels still match their source hashes. The external edits were preserved and recorded in `validation/reference_concurrent_changes.json`; see `IMPLEMENTATION_REPORT.md` for the audit chronology. The original before-edit baseline is not reset to hide them.

The MIT notices in `LICENSE` and `sources/FastGRPO/LICENSE` are preserved. The original source snapshot is test-only. `legacy_pretraining/third_party/SpecForge/LICENSE` preserves that archived tool's attribution. Original source manifests/commit attribution remain in `sources/FastGRPO/SOURCE_MANIFEST.json`. Full hashes, original destination hashes, audited reference hashes, retired paths, local destination hashes and the mapping below are machine-readable in `PORT_SOURCES.json`.

## Actual call graph audited

`train_qwen25_3b.sh` → `scripts/launch/train_model.sh` → shared/3B env → model/tokenizer, LoRA, draft and optimizer creation in `grpo_speculative.py` → `specualtive_generate` dispatch → baseline `fastgrpo_generate` or `opd_generate` → DraftModel/cache/tree/sampler/verification/history → `fastgrpo_training.training_draft_model`, analytical projector gradient application, AdamW draft boundary → `compute_target_loss` and target AdamW boundary → checkpoint state. The reference's existing policy-lag branch regenerates Fresh rollouts; it was audited but was NOT ported as the new A1 protocol.

Also traced: `METHOD_OPD_REFLEX.md`, `FASTGRPO_REWRITE.md`, data loaders/collators, rewards, model-loading and pretraining export, dynamic/static KV APIs, attention masks, finite/strict sampler, one-hop frontier selection, coarsened KL union/tail, projected proposal correction, async feedback ordering, optimizer groups, source parity and integration tests. The source math prompt is preserved. Gold parsing now rejects empty/unparseable labels rather than awarding reward 1; nonempty exact-text fallback is explicitly recorded. The upstream target trainer's decode/retokenize path and separately sorted reward arrays are replaced by records carrying actual accepted token IDs with their own aligned rewards/masks.

The old destination audit covered its `grpo_speculative.py`, `policy_lag_analysis.py`, protocol helper, LK/ReflexV5 kernels, EAGLE3/SpecForge model/supervision path, rollout merge, DAPO preparation, launchers, V5 docs and their transitive imports/tests. Those active algorithms/tests/docs are removed or rewritten. Necessary unrelated SpecForge pretraining tooling was moved intact into `legacy_pretraining/`; production `train_draft.py` is ported for future compatible FastGRPO pretraining. Old experiment names only forward to the new launcher. Legacy args and model-family mismatch fail rather than selecting another algorithm.

## Ported files

The SHA256 column is the original source hash. Destination hashes and exact exceptions are in `PORT_SOURCES.json`.

| Reference source | Local destination | Source SHA256 | Changes |
|---|---|---|---|
| `helper/opd_attention.py` | `helper/opd_attention.py` | `b7ae3e3c41c73c1763930fd172074540e26b2a3a6c028ca79cea97b8fd3071f7` | none |
| `helper/drift_metrics.py` | `helper/drift_metrics.py` | `3bef046c88aebe97c8e86c66e502f5d88ca1c3783f7783cca46e509d433f678f` | none |
| `helper/step_metrics.py` | `helper/step_metrics.py` | `0597ba0a566d1da52809c2ae725af2f4b7e75f9d6412f79e60943b6441a41062` | none |
| `helper/checkpointing.py` | `helper/checkpointing.py` | `16a8f37b5faec68a820e7cb70d8e547d90d9fd5d3ab010eb4d4066ea459854cb` | none |
| `helper/opd_history.py` | `helper/opd_history.py` | `e4f10c62a132a528dab337b51bab22ca212576fb61018c95baccc0144fe9867c` | none |
| `helper/opd_optimizer.py` | `helper/opd_optimizer.py` | `7b38bd020385a2a703e4634473d7eeb94e4e8f119ce41c8fdbca464eb4ed7387` | none |
| `helper/opd_kv_kernels.py` | `helper/opd_kv_kernels.py` | `355404f3dbfcf0a2d7be3d50d15474a9a8f7bb227ab24eb3ee009e229ad415ce` | none |
| `helper/rollout_metrics.py` | `helper/rollout_metrics.py` | `fecb866a88846f1986dbcc1f88295277b0f8554c618835f95d4f36d3d9d3923d` | none |
| `helper/specualtive_generate.py` | `helper/specualtive_generate.py` | `db187143ed4f7d79933435c40aecee2fc7f0e75a6c47a1745e9e2a08a2124313` | none |
| `helper/fastgrpo_model.py` | `helper/fastgrpo_model.py` | `cab14c5b933937627970c16a74eca935f5b7e116740438d5f624ddfaf585aea6` | none |
| `helper/tree_kernels.py` | `helper/tree_kernels.py` | `cb2bf42fd6128945ccf5ebd51c77d819648e3fea9cd53eeedc4f623614f53248` | none |
| `helper/opd_sampling.py` | `helper/opd_sampling.py` | `04d53d765fe4884aec5403ec4ec1ce869bab314d3fc577e63f36077ebc2bf5e9` | none |
| `helper/rollout_history.py` | `helper/rollout_history.py` | `903ebbb740f8595a6bb77ef4fae3def916f7d9e1ce3d1a35b1dccc0a6c936a7f` | none |
| `helper/method_config.py` | `helper/method_config.py` | `a9c8aea52278111a98e048f75332367c1fee931f6511d371be98bc46d79a3ccb` | none |
| `helper/opd_scheduling.py` | `helper/opd_scheduling.py` | `afe739c27f5f1ce32b3d448a8f39345ffd00d34376dbc34741ca6be1bcc4b3cb` | none |
| `helper/opd_attention_kernels.py` | `helper/opd_attention_kernels.py` | `afefad13dead7020885843ccde9aec84e509b7696e27cb115629000ee0b7089f` | none |
| `helper/pretrain_data.py` | `helper/pretrain_data.py` | `181e4dfb791d6cbd8125aaf4ace3a02f1654e116e2203d05cf7531e7d5e7af08` | none |
| `helper/opd_static_cache.py` | `helper/opd_static_cache.py` | `3d048c93cd70a0556acdf522df5cb55a2d050b226853e0456bc73f568ff64d98` | none |
| `helper/rewards.py` | `helper/rewards.py` | `4a14324812d999a31e8b414cd8dca33d98c275876e7b81ca1e0847874ae55f30` | Remove unparseable-gold reward=1; validate/cached gold parsing and explicit numeric LaTeX grouping normalization |
| `helper/__init__.py` | `helper/__init__.py` | `13a9095f8cbb5cab149c508822e2a5021dd033f3e4e3665c44d89fc70abd90cb` | none |
| `helper/opd_profiles.py` | `helper/opd_profiles.py` | `feacabb631e3ee1a87d4fde179ab64e9ad4308465e28e47b34b2ffb2d5092a1c` | none |
| `helper/opd_reflex_kernels.py` | `helper/opd_reflex_kernels.py` | `65fdb32878ec1aec9b3c4c83a105d9f938f7d9512aceed37ed8e2bfefd01259b` | none |
| `helper/tree_verification.py` | `helper/tree_verification.py` | `70abc46dd67af397d8288695f992b97d819cf29b73409c9dfdd333e39adeec29` | none |
| `helper/shared_rollout.py` | `helper/shared_rollout.py` | `8ec3ba1c4ff27d4f5b32ba7cdc1a97cbd7a3cb58cf0be3ad8c06fcae9eba38d8` | none |
| `helper/fastgrpo_generate.py` | `helper/fastgrpo_generate.py` | `dd9495d765b9669a6b673e61e13e2af6f073b4140b00adb1bf42c4bab64efd93` | Opt-in exact evaluation cap and compact teacher trace; fix prefill EOS termination/row compaction, unchanged non-EOS tree arithmetic |
| `helper/modeling_draft.py` | `helper/modeling_draft.py` | `5887fdbdac155af5c01c99c80249cd44baa8e5f9136fd5915aaf325047d21df4` | none |
| `helper/fastgrpo_training.py` | `helper/fastgrpo_training.py` | `bbc0971609b7ac0e4961921155ff844faec8de428303fda7e76882fa3b7a9359` | Correct padded last-token mask and empty-example normalization; stable chunked soft CE/GRPO logps, uniform final-microbatch accumulation |
| `helper/opd_generate.py` | `helper/opd_generate.py` | `d078ad2b23680c83a18e3cbb35ff7c4b2071a7577ed9404b8c9413a4d69da806` | Opt-in exact evaluation cap; fix prefill EOS termination/row compaction, unchanged proposal/feedback kernels |
| `helper/opd_reflex.py` | `helper/opd_reflex.py` | `cb6041e45aa10109d4280c93f03ffe27fdaac334d2854b23ea4e25a49917e07a` | none |
| `helper/environment_checks.py` | `helper/environment_checks.py` | `b1406dc5575937b3084ca85f993a8f8dbd483d3256f1aefd25fb8008decbe635` | none |
| `helper/transformers_compat.py` | `helper/transformers_compat.py` | `d373f2043f39ca5d7fe990e365a9c17a7ab5d29720b6c113c61cd0b1baa78b3c` | none |
| `helper/get_QAs.py` | `helper/get_QAs.py` | `72974e50046cd70fe956a00a787fdaa070bce36110ee239a7d4ecd94dcfd58d2` | Retain byte-identical math prompt; retire non-SimpleLR loaders and delegate strict official split validation |
| `sources/FastGRPO/train_draft.py` | `sources/FastGRPO/train_draft.py` | `b7e7d5f84a547e52cc59868a3bf95e96b438aff1c878ee7b29227e76acc89629` | none |
| `sources/FastGRPO/LICENSE` | `sources/FastGRPO/LICENSE` | `0530078c37f1ed2dc2bacbd92dbea826e8a4ed895978a7f421604983191ac520` | none |
| `sources/FastGRPO/README.md` | `sources/FastGRPO/README.md` | `d2012e44ab537ea51cf7200302d567075636426905a3710aa0286d36bb5dd3f2` | none |
| `sources/FastGRPO/SOURCE_MANIFEST.json` | `sources/FastGRPO/SOURCE_MANIFEST.json` | `ce49214df7b9d9a3b8a04c74251d722181f58aff1dba3cbff481cef02c1197e4` | none |
| `sources/FastGRPO/requirements.txt` | `sources/FastGRPO/requirements.txt` | `b66dd94375011a7be2d7becc5a0397654a67ea07e1e28a293fe766de4577b853` | none |
| `sources/FastGRPO/grpo_speculative.py` | `sources/FastGRPO/grpo_speculative.py` | `62bd4428a869db40ad4a8252fb895defd9bbeb33b7deffe486dc43be393607d0` | none |
| `sources/FastGRPO/helper/specualtive_generate.py` | `sources/FastGRPO/helper/specualtive_generate.py` | `7a129d33f7eb4b9e484784b91f3dabfe1797577053705aa1176cc6bf59528765` | none |
| `sources/FastGRPO/helper/rewards.py` | `sources/FastGRPO/helper/rewards.py` | `f871872c58fc3d1eef5da261b78f165bb5db0d6233d3b07de81dd600f59d7d43` | none |
| `sources/FastGRPO/helper/modeling_draft.py` | `sources/FastGRPO/helper/modeling_draft.py` | `5887fdbdac155af5c01c99c80249cd44baa8e5f9136fd5915aaf325047d21df4` | none |
| `sources/FastGRPO/helper/get_QAs.py` | `sources/FastGRPO/helper/get_QAs.py` | `aab0fe8e541f83b754910ebf3ab2e5dd935405436192733b39b46e4e0b11d0b5` | none |
| `LICENSE` | `LICENSE` | `0530078c37f1ed2dc2bacbd92dbea826e8a4ed895978a7f421604983191ac520` | none |
| `train_draft.py` | `train_draft.py` | `9bd0bb45684ccdc0ac1074fe60e3df71928e4a3fc9b168900c23c18c76590232` | none |
| `requirements.txt` | `requirements.txt` | `373731e733272f3eaab127a466a8509e425797901d1f66d4173cd649c73d1074` | none |
| `scripts/tune_opd_proposals.py` | `scripts/tune_opd_proposals.py` | `5a6fe3d0cace3344e7ca27c1a6e03b9a59955af08e868e0a1a78f2508506dbda` | none |
| `scripts/resolve_opd_profile.py` | `scripts/resolve_opd_profile.py` | `57b148eae2f2f0b2a4e4cb632c08075a8fc590759ee7a439d049e5a3a4403399` | none |
| `tests/test_opd_memory_revision.py` | `tests/test_opd_memory_revision.py` | `d4ac0ab1287008789b5a718ab4c0ffe0fe2e45454869cbb6a19c70cebcffdae8` | none |
| `tests/test_opd_last_three.py` | `tests/test_opd_last_three.py` | `c020c3b62514a7fb4ef1e15ab70b046dc7c7182cd35ad0eaa85c4a7f333a4bca` | none |
| `tests/test_fastgrpo_rewrite.py` | `tests/test_fastgrpo_rewrite.py` | `b18939043b4c427bf92b68f192d04cc0042e9a46a7405715eb9b9d26f3fdec80` | Preserve valid upstream objective parity case; independent tests expose/fix padding and final accumulation defects |
| `tests/test_opd_sampling_finite.py` | `tests/test_opd_sampling_finite.py` | `1a06c2f51ba445cdf8f2ded5e4fb8a3c931703cf6ec1bf9fc230d6eab8593171` | none |
| `tests/test_opd_reflex.py` | `tests/test_opd_reflex.py` | `164cd6e779bd81aa10541709c0626be2984f982ceb4dca9f662d27e270eb3381` | none |
| `tests/test_checkpointing.py` | `tests/test_checkpointing.py` | `d7530ab8d5785c430c1572e6b9e286765ea0f4803db525ff6c9d1b91346b828e` | none |
| `tests/test_opd_revision.py` | `tests/test_opd_revision.py` | `7f3249bc074925501c373dc3c45b8d7891b6f062603c7a0b53e79aa8cdc5cefd` | reference checkpoint AST tests use local test-only function fixture |
| `tests/test_opd_profiles.py` | `tests/test_opd_profiles.py` | `f799ef67e92103743c2ae3d0f402356cc56fc4821f4ef483c6a540df4ad4808e` | none |
| `tests/test_opd_final_optimization.py` | `tests/test_opd_final_optimization.py` | `1671fddade032211806fcab2ef4882ba8f56443b04013bc47a3ca58ad5cca4d0` | none |
| `grpo_speculative.py:checkpoint functions` | `tests/references/reference_checkpoint.py` | `4ea9b441c81250587565d1c97904cb4c7290a09103c5aa1e9256ba1724e51e2f` | AST extraction of generic checkpoint functions into test-only fixture |

## New protocol files

`run_policy_lag_motivation.py`, `run_policy_lag_motivation.sh`, `configs/qwen25_3b/simplelr_a1_a2.json`, `teacher_relabel.py`, `motivation/{config,data,state,compatibility,runtime,runner,metrics}.py`, `helper/generation_edges.py`, `tests/test_motivation_*.py`, `tests/test_teacher_relabel.py`, `tests/test_correctness_fixes.py`, `scripts/{audit_port.py,validate_b200.sh,verify_execution.py}`, `DESIGN_A1_A2.md`, README and validation/report artifacts. `grpo_speculative.py`, `policy_lag_analysis.py` and retained old launcher names are small compatibility forwarders, not parallel algorithms.

Five ported runtime files differ from source: the strict SimpleLR loader, rewards, numerically stable mask-correct training objectives, and the two generators with evaluation caps, compact teacher tracing and prefill-EOS row handling. Verified mathematical fixes are documented in CORRECTNESS_VALIDATION.md and DESIGN_A1_A2.md; non-EOS proposal/feedback paths retain source parity coverage. All OPD proposal, finite sampling, feedback, A optimizer, tree, attention and KV modules remain unchanged copies. The local helper/generation_edges.py only handles immediate prefill EOS. The full list of original retired active paths follows.

- `helper/eagle3_supervision.py`
- `helper/drift_metrics.py`
- `helper/fast_lk_reflex_kernels.py`
- `helper/fast_lk_reflex.py`
- `helper/policy_lag_protocol.py`
- `helper/rollout_merge.py`
- `helper/specualtive_generate.py`
- `helper/rollout_history.py`
- `helper/rewards.py`
- `helper/__init__.py`
- `helper/tree_verification.py`
- `helper/reflex_port.json`
- `helper/modeling_draft.py`
- `helper/sampling.py`
- `helper/eagle3_specforge.py`
- `helper/response_batches.py`
- `helper/get_QAs.py`
- `tests/test_launcher_contract.py`
- `tests/test_policy_lag_analysis.py`
- `tests/test_rollout_merge.py`
- `tests/test_reflex.py`
- `tests/test_eagle3_supervision.py`
- `tests/test_policy_lag_protocol.py`
- `tests/test_policy_lag_v5_exports.py`
- `tests/test_response_batches.py`
- `tests/test_reflex_decoder.py`
- `scripts/generate_eagle3_config.py`
- `scripts/prepare_dapo_policy_lag.py`
- `scripts/launch/train_model.sh`
- `scripts/launch/pretrain_model.sh`
- `train_fastgrpo.sh`
- `run_policy_lag_analysis_b200.sh`
- `train_qwen25_1p5b.sh`
- `train_llama31_8b.sh`
- `train_qwen25_14b.sh`
- `train_qwen25_3b.sh`
- `run_b200_policy_lag_pipeline.sh`
- `train_qwen25_7b.sh`
- `run_policy_lag_analysis.sh`
- `README_B200_POLICY_LAG.md`
- `README.md`
- `README_POLICY_LAG.md`
- `README_B200.md`
- `POLICY_LAG_V5.md`
- `DEPENDENCIES_POLICY_LAG.md`
- `requirements-policy-lag.txt`
- `policy_lag_analysis.py`
- `grpo_speculative.py`

## Verification commands

```bash
python scripts/audit_port.py
python scripts/audit_port.py --reference ../SpecNaacl
python -m pytest -q
```

The reference argument is only an optional read-only development audit; it is never used by the experiment. Current validation logs and precise hardware/model/checkpoint limits are recorded in `CORRECTNESS_VALIDATION.md`; `IMPLEMENTATION_REPORT.md` describes the historical initial port.

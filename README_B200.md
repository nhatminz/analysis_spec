# FastGRPO B200 Launchers

This folder stays close to the upstream `yedaotian9/FastGRPO` layout, with B200 launch scripts added around the original `train_draft.py` and `grpo_speculative.py`.

## Output Layout

- Draft pretrain checkpoints: `outputs/fastgrpo/pretrain/<model_key>/step*.pth`
- Draft pretrain logs: `logs/fastgrpo/pretrain/<model_key>/<run_name>/`
- GRPO train outputs: `outputs/fastgrpo/train/<model_key>/<run_name>/`
- GRPO train logs: `logs/fastgrpo/train/<model_key>/<run_name>/`

## Model Scripts

Each model has a pretrain script and a train script:

- `bash fastgrpo/pretrain_qwen25_1p5b.sh`
- `bash fastgrpo/train_qwen25_1p5b.sh`
- `bash fastgrpo/pretrain_qwen25_3b.sh`
- `bash fastgrpo/train_qwen25_3b.sh`
- `bash fastgrpo/pretrain_qwen25_7b.sh`
- `bash fastgrpo/train_qwen25_7b.sh`
- `bash fastgrpo/pretrain_qwen25_14b.sh`
- `bash fastgrpo/train_qwen25_14b.sh`
- `bash fastgrpo/pretrain_llama31_8b.sh`
- `bash fastgrpo/train_llama31_8b.sh`

## Common Overrides

Examples:

```bash
DATASET=gsm8k TRAIN_DATA_FRACTION=0.4 bash fastgrpo/train_qwen25_7b.sh
DATASET=simplelr TRAIN_DATA_FRACTION=0.4 bash fastgrpo/train_qwen25_7b.sh
DATASET=dapo TRAIN_DATA_FRACTION=0.4 bash fastgrpo/train_qwen25_7b.sh
```

Use `DRAFT_ADAPTER=/path/to/stepXXXX.pth` to force a specific draft checkpoint. If it is omitted, the train launcher picks the newest `step*.pth` from `outputs/fastgrpo/pretrain/<model_key>/`.

Use `RESUME_CHECKPOINT=auto` to resume the newest interrupted training checkpoint from the current train output directory.

Use `DRY_RUN=true` to print the full command without launching training.

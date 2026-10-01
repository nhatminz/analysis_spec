#!/usr/bin/env python3
"""Generate a one-layer SpecForge EAGLE-3 config from a local target config."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ARCHITECTURE_FIELDS = (
    "vocab_size",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "hidden_act",
    "initializer_range",
    "attention_bias",
    "attention_dropout",
    "head_dim",
    "rope_theta",
    "rope_scaling",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "sliding_window",
    "use_sliding_window",
    "max_window_layers",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draft-vocab-size", type=int, default=16000)
    args = parser.parse_args()

    target_dir = Path(args.target_model_path).expanduser().resolve()
    source = target_dir / "config.json"
    output = Path(args.output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"target config not found: {source}")
    target = json.loads(source.read_text(encoding="utf-8"))
    if target.get("model_type") != "qwen2":
        raise ValueError(
            "the bundled FastGRPO runtime adapter currently supports local "
            f"Qwen2/Qwen2.5 targets, got model_type={target.get('model_type')!r}"
        )
    for key in ("vocab_size", "hidden_size", "num_hidden_layers"):
        if not isinstance(target.get(key), int) or target[key] <= 0:
            raise ValueError(f"target config requires a positive integer {key}")
    if not 0 < args.draft_vocab_size <= target["vocab_size"]:
        raise ValueError(
            f"draft vocab size must be in [1, {target['vocab_size']}], got {args.draft_vocab_size}"
        )

    draft = {key: target[key] for key in ARCHITECTURE_FIELDS if target.get(key) is not None}
    draft.update(
        {
            "architectures": ["LlamaForCausalLMEagle3"],
            "model_type": "llama",
            "num_hidden_layers": 1,
            "tie_word_embeddings": False,
            "use_cache": True,
            "pad_token_id": target.get("pad_token_id", 0),
            "draft_vocab_size": args.draft_vocab_size,
            "eagle_config": {
                "eagle_aux_hidden_state_layer_ids": [
                    1,
                    target["num_hidden_layers"] // 2 - 1,
                    target["num_hidden_layers"] - 4,
                ]
            },
        }
    )
    if min(draft["eagle_config"]["eagle_aux_hidden_state_layer_ids"]) < 0:
        raise ValueError("target has too few layers for the EAGLE-3 three-layer feature rule")

    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(draft, indent=2, sort_keys=True) + "\n"
    if output.is_file() and output.read_text(encoding="utf-8") != encoded:
        raise RuntimeError(f"refusing to overwrite a different generated config: {output}")
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "target_model_path": str(target_dir),
                "target_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "draft_config": str(output),
                "draft_vocab_size": args.draft_vocab_size,
                "feature_layers": draft["eagle_config"]["eagle_aux_hidden_state_layer_ids"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

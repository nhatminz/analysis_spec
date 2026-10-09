"""Fail closed on model/checkpoint/tokenizer incompatibility; never download/fallback."""
from pathlib import Path
from copy import deepcopy
import json
import torch
from motivation.state import file_hash

QWEN_3B = dict(model_type='qwen2', hidden_size=2048, intermediate_size=11008,
               num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2,
               vocab_size=151936, bos_token_id=151643, eos_token_id=151645, tie_word_embeddings=True)


def validate_model_config(config):
    for key, expected in QWEN_3B.items():
        if config.get(key) != expected:
            raise ValueError(f'Qwen2.5-3B-Instruct required: config.{key}={config.get(key)!r}, expected {expected!r}')
    theta = config.get('rope_theta', config.get('rope_parameters', {}).get('rope_theta'))
    if theta != 1000000.: raise ValueError('Target RoPE does not match Qwen2.5-3B')


def model_identity(model_path):
    path = Path(model_path).expanduser().resolve(strict=True)
    config = json.loads((path/'config.json').read_text()); validate_model_config(config)
    files = sorted(set(path.glob('*.safetensors')) | set(path.glob('pytorch_model*.bin')))
    if not files: raise FileNotFoundError(f'No local model weights at {path}; online fallback is disabled')
    hashes = {p.name: file_hash(p) for p in files}
    return dict(path=str(path), variant='Qwen2.5-3B-Instruct', config=config,
                config_sha256=file_hash(path/'config.json'), weight_sha256=hashes)


def validate_tokenizer(tokenizer, config):
    expected = {'<|endoftext|>':151643, '<|im_start|>':151644, '<|im_end|>':151645}
    if tokenizer.eos_token_id != 151645 or tokenizer.pad_token_id != 151643:
        raise ValueError('Qwen2.5-Instruct EOS/pad IDs do not match the reference')
    for token, index in expected.items():
        if tokenizer.convert_tokens_to_ids(token) != index: raise ValueError(f'Tokenizer mismatch for {token}')
    if not tokenizer.chat_template or '<|im_start|>' not in tokenizer.chat_template:
        raise ValueError('Qwen Instruct chat template is required')
    if max(tokenizer.get_vocab().values()) >= config['vocab_size']:
        raise ValueError('Tokenizer IDs exceed target vocabulary')
    if tokenizer.padding_side != 'left': raise ValueError('Generation must use left padding')


def validate_draft(path, target_path, target_config_path=''):
    from transformers import AutoConfig
    from helper.modeling_draft import DraftModel
    p = Path(path).expanduser().resolve(strict=True)
    if p.is_dir(): p = p/'draft.pth'
    if not p.is_file(): raise FileNotFoundError(f'FastGRPO draft.pth missing: {p}')
    companion = Path(target_config_path).expanduser().resolve(strict=True) if target_config_path else p.parent/'target_config.json'
    if not companion.is_file():
        raise FileNotFoundError(f'Checkpoint target provenance missing: {companion}; pass --draft-target-config from its pretrain run')
    saved_config = json.loads(companion.read_text()); validate_model_config(saved_config)
    current = json.loads((Path(target_path)/'config.json').read_text()); validate_model_config(current)
    for key in QWEN_3B:
        if saved_config.get(key) != current.get(key): raise ValueError(f'Checkpoint target config mismatch: {key}')
    cfg = AutoConfig.from_pretrained(target_path, local_files_only=True)
    cfg.num_hidden_layers=1; cfg.rope_scaling=None; cfg.torch_dtype=torch.bfloat16
    with torch.device('meta'): expected = DraftModel(cfg).state_dict()
    payload = torch.load(p, map_location='cpu', weights_only=True)
    if not isinstance(payload,dict) or 'draft_model' not in payload:
        raise ValueError('Expected FastGRPO {draft_model: state_dict}; EAGLE3/SpecForge exports cannot be used')
    state = payload['draft_model']
    if set(state) != set(expected):
        raise ValueError(f'FastGRPO draft architecture mismatch: missing={sorted(set(expected)-set(state))[:8]}, '
                         f'extra={sorted(set(state)-set(expected))[:8]}')
    for name, tensor in state.items():
        if tuple(tensor.shape) != tuple(expected[name].shape):
            raise ValueError(f'Draft {name} shape {tuple(tensor.shape)} != {tuple(expected[name].shape)}')
        if not torch.isfinite(tensor).all(): raise ValueError(f'Nonfinite pretrained draft: {name}')
    # FastGRPO checkpoints carry no embedding/head: these are shared, frozen
    # target modules. The strict key set rejects compact heads/mapping exports.
    return dict(path=str(p.resolve()), sha256=file_hash(p), target_config_path=str(companion.resolve()),
                target_config_sha256=file_hash(companion), architecture='FastGRPO DraftModel, one decoder layer',
                embedding_and_head='shared frozen target; full vocabulary; no compact mapping',
                tokenizer_provenance='target-config special IDs; upstream pretrain exports no tokenizer hash')


def tokenizer_identity(path):
    p = Path(path)
    return {f.name:file_hash(f) for f in sorted(p.glob('*')) if f.is_file() and
            (f.name.startswith('tokenizer') or f.name in ('vocab.json','merges.txt','special_tokens_map.json','added_tokens.json'))}

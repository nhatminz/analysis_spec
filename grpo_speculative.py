import os
import sys
import random
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
# Pin helper imports to THIS repository even when the parent workspace has its
# own helper package or the command is launched from another working directory.
sys.path = [str(REPO_ROOT)] + [entry for entry in sys.path if entry != str(REPO_ROOT)]

import pandas as pd
from transformers import AutoTokenizer,AutoConfig,AutoModelForCausalLM,GenerationConfig
from helper.modeling_draft import Model
from helper.rewards import accuracy_reward_func , format_reward_func
from helper.get_QAs import get_test_QAs , get_train_QAs, get_QAs_from_path, select_train_subset
from helper.drift_metrics import acceptance_aligned_union_metrics
from helper.specualtive_generate import speculative_generate_in_prompt_batches
from helper.eagle3_specforge import Eagle3FastGRPOAdapter
from helper.eagle3_supervision import aligned_eagle3_row, count_eagle3_supervision
from helper.response_batches import reorder_response_fields, microbatch_index_groups
from helper.policy_lag_protocol import is_analysis_boundary, branch_spec, paired_shadow_rollout
from policy_lag_analysis import (
    PROTOCOL_VERSION,
    AnalysisIO,
    BranchSummary,
    append_boundary_results,
    cleanup_branch_checkpoints,
    mark_durable_boundaries,
    atomic_json,
    bootstrap_delta_by_prompt,
    load_completed_results,
    state_digest,
    teacher_shift_tv,
    weighted_aal,
    write_results,
)
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
from torch.utils.data import DataLoader
import numpy as np
import json
import pandas as pd
import signal
import torch
from copy import deepcopy
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType
try:
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
except ImportError:
    get_peft_model_state_dict = None
    set_peft_model_state_dict = None
from datetime import datetime
import argparse 
from statistics import mean , stdev
import pickle
import importlib.util
from tqdm.auto import tqdm

def handle_signal(signum, frame):
    print("Received signal, cleaning up...")
    if torch.cuda.is_available():
        del model
        torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def _dtype_from_name(name):
    name = str(name or "auto").lower()
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name}")


def _resolve_attn_implementation(requested):
    requested = str(requested or "")
    if not requested:
        return None
    if requested == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        print(
            "Warning: attn_implementation=flash_attention_2 was requested, "
            "but flash_attn is not installed. Falling back to eager."
        )
        return "eager"
    return requested


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def _cpu_snapshot(value):
    # All GPU->CPU copies must be OUTSIDE the analysis IO timer.
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_snapshot(item) for item in value)
    return deepcopy(value)


def _prune_checkpoints(checkpoint_dir, keep_last):
    keep_last = int(keep_last or 0)
    if keep_last <= 0:
        return
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(
        [*checkpoint_dir.glob("step*.pt"), *checkpoint_dir.glob("target*.pt")],
        key=lambda p: p.stat().st_mtime,
    )
    for old_path in checkpoints[:-keep_last]:
        old_path.unlink(missing_ok=True)


def _target_lora_state_dict(target_model):
    if get_peft_model_state_dict is not None:
        return get_peft_model_state_dict(target_model)
    return target_model.state_dict()


def _load_target_lora_state_dict(target_model, state_dict):
    if set_peft_model_state_dict is not None:
        set_peft_model_state_dict(target_model, state_dict)
    else:
        target_model.load_state_dict(state_dict, strict=False)


def save_training_checkpoint(
    checkpoint_dir,
    *,
    model,
    optimizer_target,
    optimizer_draft,
    epoch,
    next_batch,
    step,
    used_items,
    draft_step,
    draft_accumulated_step,
    batch_data,
    keep_last,
    target_optimizer_steps=None,
):
    checkpoint_dir = Path(checkpoint_dir)
    state = {
        "format": "fastgrpo_grpo_checkpoint_v1",
        "epoch": int(epoch),
        "next_batch": int(next_batch),
        "step": int(step),
        "target_optimizer_steps": None if target_optimizer_steps is None else int(target_optimizer_steps),
        "used_items": int(used_items),
        "draft_step": int(draft_step),
        "draft_accumulated_step": int(draft_accumulated_step),
        "target_lora": _target_lora_state_dict(model.target_model),
        "draft_model": model.draft_model.state_dict(),
        "optimizer_target": optimizer_target.state_dict(),
        "optimizer_draft": optimizer_draft.state_dict(),
        "batch_data": batch_data,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "rng_state": torch.random.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    checkpoint_label = (
        f"target{int(target_optimizer_steps)}" if target_optimizer_steps is not None
        else f"step{int(step)}"
    )
    checkpoint_path = checkpoint_dir / f"{checkpoint_label}_epoch{int(epoch) + 1}_batch{int(next_batch)}.pt"
    _atomic_torch_save(state, checkpoint_path)
    _atomic_torch_save(state, checkpoint_dir / "latest.pt")
    _prune_checkpoints(checkpoint_dir, keep_last)
    print(f"Saved FastGRPO checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_training_checkpoint(path, *, model, optimizer_target, optimizer_draft):
    # This is our own trusted training checkpoint, not a weights-only export:
    # it includes Python/NumPy RNG and optimizer/trajectory state.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.draft_model.load_state_dict(checkpoint["draft_model"])
    _load_target_lora_state_dict(model.target_model, checkpoint["target_lora"])
    optimizer_target.load_state_dict(checkpoint["optimizer_target"])
    optimizer_draft.load_state_dict(checkpoint["optimizer_draft"])
    if checkpoint.get("python_rng_state") is not None:
        random.setstate(checkpoint["python_rng_state"])
    if checkpoint.get("numpy_rng_state") is not None:
        np.random.set_state(checkpoint["numpy_rng_state"])
    if checkpoint.get("rng_state") is not None:
        torch.random.set_rng_state(checkpoint["rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint


parser = argparse.ArgumentParser(description="Training configuration")

parser.add_argument('--model_dir',type=str)
parser.add_argument('--adapter_path',type=str)
parser.add_argument('--draft_backend', type=str, default='legacy', choices=['legacy', 'eagle3'])
parser.add_argument('--draft_config', type=str, default='')
parser.add_argument('--draft_initialization_mode', type=str, default='pretrained', choices=['pretrained', 'random'])
parser.add_argument('--vocab_mapping', type=str, default='')
parser.add_argument('--eagle_feature_layers', type=str, default='',
                    help='Comma-separated decoder layer ids; empty uses the SpecForge EAGLE-3 rule.')
parser.add_argument('--eagle_ttt_length', type=int, default=7)
parser.add_argument('--eagle_lk_loss_type', type=str, default='', choices=['', 'lambda', 'alpha', 'tv'])
parser.add_argument('--eagle_kl_scale', type=float, default=1.0)
parser.add_argument('--eagle_kl_decay', type=float, default=1.0)
parser.add_argument('--dtype', type=str, default='auto', choices=['auto', 'bf16', 'fp16', 'fp32'])
parser.add_argument('--attn_implementation', type=str, default='')
parser.add_argument('--temperature',type=float,default=1.0)
parser.add_argument('--top_p',type=float,default=0.95)
parser.add_argument('--accumulation_steps', type=int, default=2, help='Gradient accumulation steps for target model')
parser.add_argument('--draft_accumulation_steps', type=int, default=1, help='Gradient accumulation steps for draft model')
parser.add_argument('--target_lr', type=float, default=1e-6, help='Learning rate for target model')
parser.add_argument('--draft_lr', type=float, default=1e-4, help='Learning rate for draft model')
parser.add_argument('--is_train_draft', type=lambda x: x.lower() == 'true', default=True, help='Whether to train the draft model (True/False)')
parser.add_argument('--model_type', type=str, default='Qwen2___5-Math-7B', help='Version name for saving checkpoints')
parser.add_argument('--train_option',type=str,default="simplelr_abel_level3to5")
parser.add_argument('--dataset_path', type=str, default='')
parser.add_argument('--eval_dataset_path', type=str, default='',
                    help='Optional held-out evaluation dataset. Defaults to dataset_path only when that path has a real eval split.')
parser.add_argument('--train_split', type=str, default='train')
parser.add_argument('--eval_split', type=str, default='test')
parser.add_argument('--load_lora_path',type=str,default="")
parser.add_argument('--batch_size',type=int,default=4)
parser.add_argument('--version_name',type=str,default='normal')
parser.add_argument('--num_epochs',type=int,default=10)
parser.add_argument('--sample_num',type=int,default=100)
parser.add_argument('--train_data_fraction', type=float, default=0.4,
                    help='Fraction of the loaded train split to use. Applied to any train_option dataset.')
parser.add_argument('--train_subset_seed', type=int, default=42,
                    help='Seed for the deterministic train subset selection.')
parser.add_argument('--max_train_samples', type=int, default=0,
                    help='Optional hard cap after train_data_fraction, useful for smoke/debug runs.')
parser.add_argument('--grpo_iteration_num',type=int,default=1)
parser.add_argument('--repeated_generate_nums',type=int,default=8)
parser.add_argument('--beta',type=float,default=0.01)
parser.add_argument('--epsilon',type=float,default=0.1)
parser.add_argument('--max_length',type=int,default=2048)
parser.add_argument('--max_prompt_length', type=int, default=2048,
                    help='Maximum prompt tokens before rollout; kept separate from max generated sequence length.')
parser.add_argument('--max_training_padding_gap',type=int,default=256)
parser.add_argument('--max_training_token',type=int,default=3072)
parser.add_argument('--logps_chunk_size', type=int, default=256,
                    help='Sequence chunk size for token-logprob computation. Lower values reduce peak VRAM.')
parser.add_argument('--verification_capacity', type=int, default=160)
parser.add_argument('--max_draft_token_length', type=int, default=5)
parser.add_argument('--max_draft_k', type=int, default=8)
parser.add_argument('--max_verification_num', type=int, default=160)
parser.add_argument('--min_draft_token_length', type=int, default=3)
parser.add_argument('--draft_token_length_c', type=float, default=0.75)
parser.add_argument('--statistical_time', type=lambda x: x.lower() == 'true', default=False,
                    help='Collect detailed speculative timing counters. False avoids extra CUDA synchronizations.')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--persistent_workers', default=True)
parser.add_argument('--log_file', type=str, required=True,
                    help="Full path to training log file, e.g., /path/to/train.log")
parser.add_argument('--summary_file', type=str, default='',
                    help="Optional summary JSON path. Defaults to summary.json next to log_file.")
parser.add_argument('--saved_model_dir', type=str, required=True,
                    help="Directory to save trained target adapter/model checkpoints")
parser.add_argument('--saved_draft_model_dir', type=str, required=True,
                    help="Directory to save trained draft model checkpoints")
parser.add_argument('--saved_statistics_dir', type=str, required=True,
                    help="Directory to save statistics of generated sequence lengths.")
parser.add_argument('--checkpoint_dir', type=str, default='')
parser.add_argument('--save_checkpoint_steps', type=int, default=0)
parser.add_argument('--keep_last_checkpoints', type=int, default=3)
parser.add_argument('--resume_checkpoint', type=str, default='')
parser.add_argument('--append_log', default='',
                    help='Append JSONL explicitly. Empty preserves legacy behavior (append when resuming).')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--reset_rng_on_resume', default=False,
                    help='Reset RNGs to --seed after loading a checkpoint; use true for paired trace runs.')
parser.add_argument('--max_grpo_steps', type=int, default=0,
                    help='Analysis: absolute target optimizer-step limit (also on resume). Otherwise new GRPO steps; 0 disables.')
parser.add_argument('--drift_topk', type=int, default=16,
                    help='Per-distribution top-k used by the bidirectional sparse-union drift metric.')
parser.add_argument('--drift_temperature', type=float, default=1.0,
                    help='Temperature used for sparse TV and forward-KL logging.')
parser.add_argument('--drift_row_chunk_size', type=int, default=32,
                    help='Valid token rows per sparse-drift chunk; lower values reduce peak VRAM.')
parser.add_argument('--draft_lr_multiplier', type=float, default=1.0,
                    help='Explicit FastGRPO ablation multiplier applied after optimizer resume; 1.0 is the fair baseline.')
parser.add_argument('--policy_lag_output_dir', type=str, default='')
parser.add_argument('--analysis_boundaries', type=str, default='')
parser.add_argument('--analysis_interval', type=int, default=0)
parser.add_argument('--analysis_eval_prompts', type=int, default=8,
                    help='Prompts in the next real GRPO rollout; must equal --batch_size.')
parser.add_argument('--analysis_eval_batch_size', type=int, default=8,
                    help='Maximum prompts per speculative decoder call, not a held-out split.')
parser.add_argument('--analysis_training_token_budget', type=int, default=0)
parser.add_argument('--analysis_draft_update_steps', type=int, default=1)
parser.add_argument('--analysis_bootstrap_samples', type=int, default=2000)
parser.add_argument('--analysis_resume', default='true')
parser.add_argument('--analysis_keep_branch_checkpoints', default='0')
parser.add_argument('--reflex_backend', choices=['triton', 'torch', 'auto'], default='triton')
parser.add_argument('--reflex_feedback_scope', choices=['root', 'visited_path'], default='root')
parser.add_argument('--reflex_proposal_strategy', choices=['fused', 'sort', 'hybrid', 'torch'], default='fused')
parser.add_argument('--reflex_correction_strategy', choices=['serial', 'parallel', 'tiled'], default='serial')
parser.add_argument('--reflex_feedback_strategy', choices=['serial', 'parallel'], default='serial')
parser.add_argument('--reflex_feature_strategy', choices=['auto', 'triton', 'torch'], default='auto')
parser.add_argument('--reflex_update_stream', default='1')
parser.add_argument('--reflex_feature_dim', type=int, default=8)
parser.add_argument('--reflex_lr', type=float, default=0.05)
parser.add_argument('--reflex_weight_decay', type=float, default=0.0)
args = parser.parse_args()
num_epochs=args.num_epochs
sample_num=args.sample_num
train_data_fraction=args.train_data_fraction
train_subset_seed=args.train_subset_seed
max_train_samples=args.max_train_samples
grpo_iteration_num=args.grpo_iteration_num
repeated_generate_nums=args.repeated_generate_nums
beta=args.beta
epsilon=args.epsilon
max_length=args.max_length
max_prompt_length=args.max_prompt_length
max_training_padding_gap=args.max_training_padding_gap
max_training_token=args.max_training_token
logps_chunk_size=max(1, args.logps_chunk_size)
verification_capacity=args.verification_capacity
max_draft_token_length=args.max_draft_token_length
max_draft_k=args.max_draft_k
max_verification_num=args.max_verification_num
min_draft_token_length=args.min_draft_token_length
draft_token_length_c=args.draft_token_length_c
statistical_time=args.statistical_time
num_workers=args.num_workers
persistent_workers=_as_bool(args.persistent_workers)
batch_size = args.batch_size
accumulation_steps = args.accumulation_steps
draft_accumulation_steps = args.draft_accumulation_steps
target_lr = args.target_lr
draft_lr = args.draft_lr
is_train_draft = args.is_train_draft
model_type = args.model_type
model_dir = args.model_dir
adapter_path = args.adapter_path
temperature = args.temperature
top_p = args.top_p
version_name = args.version_name
log_file = args.log_file
summary_file = args.summary_file or os.path.join(os.path.dirname(log_file), "summary.json")
saved_model_dir = args.saved_model_dir
saved_draft_model_dir = args.saved_draft_model_dir
saved_statistics_dir = args.saved_statistics_dir
checkpoint_dir = args.checkpoint_dir or os.path.join(os.path.dirname(saved_model_dir), "checkpoints")
save_checkpoint_steps = int(args.save_checkpoint_steps or 0)
keep_last_checkpoints = int(args.keep_last_checkpoints or 0)
resume_checkpoint = args.resume_checkpoint
append_log = bool(resume_checkpoint) if str(args.append_log) == '' else _as_bool(args.append_log)
trace_seed = int(args.seed)
reset_rng_on_resume = _as_bool(args.reset_rng_on_resume)
_seed_everything(trace_seed)
max_grpo_steps = max(0, int(args.max_grpo_steps))
drift_topk = max(1, int(args.drift_topk))
drift_temperature = float(args.drift_temperature)
drift_row_chunk_size = max(1, int(args.drift_row_chunk_size))
draft_lr_multiplier = float(args.draft_lr_multiplier)
policy_lag_output_dir = args.policy_lag_output_dir
analysis_boundaries = {
    int(item) for item in args.analysis_boundaries.split(',') if item.strip()
}
analysis_interval = max(0, int(args.analysis_interval))
analysis_enabled = bool(policy_lag_output_dir) and bool(analysis_boundaries or analysis_interval)
analysis_eval_prompts = int(args.analysis_eval_prompts)
analysis_eval_batch_size = int(args.analysis_eval_batch_size)
analysis_io = AnalysisIO()
analysis_keep_branch_checkpoints = _as_bool(args.analysis_keep_branch_checkpoints)
reflex_config = {
    'reflex_backend': args.reflex_backend,
    'reflex_feedback_scope': args.reflex_feedback_scope,
    'reflex_proposal_strategy': args.reflex_proposal_strategy,
    'reflex_correction_strategy': args.reflex_correction_strategy,
    'reflex_feedback_strategy': args.reflex_feedback_strategy,
    'reflex_feature_strategy': args.reflex_feature_strategy,
    'reflex_update_stream': _as_bool(args.reflex_update_stream),
    'reflex_feature_dim': int(args.reflex_feature_dim),
    'reflex_lr': float(args.reflex_lr),
    'reflex_weight_decay': float(args.reflex_weight_decay),
    'reflex_seed': trace_seed,
}
if analysis_enabled:
    if analysis_eval_prompts <= 0 or analysis_eval_batch_size <= 0:
        raise ValueError('analysis evaluation prompt count and batch size must be positive')
    if int(args.analysis_training_token_budget) <= 0:
        raise ValueError('analysis draft supervision token budget must be positive')
    if batch_size != analysis_eval_prompts:
        raise ValueError(
            'next-real-rollout analysis requires --batch_size == --analysis_eval_prompts; '
            'the evaluated prompt batch must also be used for GRPO'
        )
if analysis_enabled:
    analysis_root = Path(policy_lag_output_dir)
    source_sha256 = {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
        for name in (
            'grpo_speculative.py', 'helper/specualtive_generate.py',
            'helper/eagle3_supervision.py', 'helper/response_batches.py',
            'helper/rollout_merge.py', 'helper/eagle3_specforge.py',
            'helper/fast_lk_reflex.py', 'helper/fast_lk_reflex_kernels.py',
            'helper/sampling.py', 'helper/tree_verification.py', 'helper/rollout_history.py',
            'helper/policy_lag_protocol.py', 'helper/reflex_port.json',
            'policy_lag_analysis.py', 'scripts/prepare_dapo_policy_lag.py',
            'third_party/SpecForge/specforge/modeling/target/target_head.py',
            'third_party/SpecForge/specforge/algorithms/eagle3/model.py',
        )
    }
    protocol_path = analysis_root / 'protocol.json'
    protocol = {
        'format': PROTOCOL_VERSION,
        'branches': ['stale', 'fresh', 'reflex'],
        'reflex_draft': 'phi_stale',
        'reflex_state_scope': 'zero_per_response_per_rollout; destroyed_after_rollout',
        'reflex_config': reflex_config,
        'policy_step_limit_semantics': 'absolute_target_optimizer_steps',
        'evaluation_scope': 'next_real_grpo_rollout',
        'evaluation_target': 'theta_after_update',
        'fresh_rollout_draft': 'phi_base',
        'main_trajectory': 'stale_branch_rollout_used_for_grpo',
        'pairing': 'same_next_batch_and_captured_training_rng',
        'data_order': 'independent_per_epoch_generator_seed_plus_epoch',
        'teacher_alignment': 'specforge_target_head_preprocess_shift_final_hidden_v1',
        'response_advantage_alignment': 'stable_length_permutation_v1',
        'evaluation_prompt_count': analysis_eval_prompts,
        'evaluation_decoder_batch_size': analysis_eval_batch_size,
        'responses_per_prompt': repeated_generate_nums,
        'generation_parameters': {
            'max_length': max_length, 'max_prompt_length': max_prompt_length,
            'temperature': temperature, 'top_p': top_p,
            'verification_capacity': verification_capacity,
            'max_verification_num': max_verification_num,
            'max_draft_token_length': max_draft_token_length, 'max_draft_k': max_draft_k,
            'min_draft_token_length': min_draft_token_length, 'draft_token_length_c': draft_token_length_c,
        },
        'target_lr': float(target_lr),
        'requested_training_token_budget': int(args.analysis_training_token_budget),
        'draft_optimizer_steps_per_branch': int(args.analysis_draft_update_steps),
        'requested_draft_lr': float(draft_lr),
        'target_max_training_token': int(max_training_token),
        'target_model_path': os.path.realpath(model_dir),
        'draft_checkpoint_path': os.path.realpath(adapter_path),
        'draft_config_sha256': hashlib.sha256(Path(args.draft_config).read_bytes()).hexdigest(),
        'vocab_mapping_sha256': (hashlib.sha256(Path(args.vocab_mapping).read_bytes()).hexdigest()
                                 if args.vocab_mapping else None),
        'train_dataset_path': os.path.realpath(args.dataset_path),
        'train_dataset_sha256': (
            hashlib.sha256(Path(args.dataset_path).read_bytes()).hexdigest()
            if Path(args.dataset_path).is_file() else None
        ),
        'eval_dataset_sha256': (
            hashlib.sha256(Path(args.eval_dataset_path).read_bytes()).hexdigest()
            if args.eval_dataset_path and Path(args.eval_dataset_path).is_file() else None
        ),
        'source_sha256': source_sha256,
    }
    if protocol_path.is_file():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise RuntimeError(f'incompatible policy-lag protocol in {protocol_path}; use a new OUTPUT_DIR')
    elif any((analysis_root / name).exists() for name in ('summary.jsonl', 'per_response.jsonl', 'boundaries')):
        raise RuntimeError(f'existing policy-lag results have no v5 protocol in {analysis_root}; use a new OUTPUT_DIR')
    else:
        atomic_json(protocol_path, protocol)
    if not _as_bool(args.analysis_resume) and any(
        (analysis_root / name).exists() for name in ('summary.jsonl', 'per_response.jsonl', 'boundaries')
    ):
        raise RuntimeError(
            f'ANALYSIS_RESUME=false requires a new OUTPUT_DIR; existing results found in {analysis_root}'
        )
analysis_completed = set()
analysis_per_response = []
analysis_summaries = []
if policy_lag_output_dir and _as_bool(args.analysis_resume):
    analysis_completed, analysis_per_response, analysis_summaries = load_completed_results(
        Path(policy_lag_output_dir)
    )
if analysis_enabled and args.draft_backend != 'eagle3':
    raise ValueError('policy-lag analysis requires --draft_backend=eagle3')
if analysis_enabled and draft_accumulation_steps != 1:
    raise ValueError('paired policy-lag branches require --draft_accumulation_steps=1')
if analysis_enabled and grpo_iteration_num != 1:
    raise ValueError('policy-lag boundary pairing currently requires --grpo_iteration_num=1')
if int(args.analysis_draft_update_steps) <= 0:
    raise ValueError('--analysis_draft_update_steps must be positive')
if drift_temperature <= 0.0:
    raise ValueError('--drift_temperature must be positive')
if draft_lr_multiplier <= 0.0:
    raise ValueError('--draft_lr_multiplier must be positive')
fastgrpo_ablation = not np.isclose(draft_lr_multiplier, 1.0)
model_torch_dtype = _dtype_from_name(args.dtype)
attn_impl = _resolve_attn_implementation(args.attn_implementation)

if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)
if not os.path.exists(saved_draft_model_dir):
    os.makedirs(saved_draft_model_dir)
if not os.path.exists(saved_statistics_dir):
    os.makedirs(saved_statistics_dir)
if checkpoint_dir:
    os.makedirs(checkpoint_dir, exist_ok=True)
if os.path.dirname(summary_file):
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)


print(datetime.now())
print(model_type,os.getenv('CUDA_VISIBLE_DEVICES'))
print("=" * 60)
print("Training & Generation Configuration")
print("=" * 60)
print(f"Model: {model_type} | Version: {version_name}")
print(f"Path: model={model_dir}, adapter={adapter_path}")
print(f"Train: epochs={num_epochs}, batch={batch_size}, "
      f"acc_steps={accumulation_steps}, draft_acc_steps={draft_accumulation_steps}")
print(f"LR: target={target_lr}, draft={draft_lr} | "
      f"Seq: max_len={max_length}, max_prompt={max_prompt_length}, "
      f"max_tokens={max_training_token}, pad_gap={max_training_padding_gap}, "
      f"logps_chunk={logps_chunk_size}")
print(f"Gen: temp={temperature}, top_p={top_p}"
      f"beta={beta}, epsilon={epsilon}")
print(f"B200/spec: dtype={args.dtype}, attn_impl={attn_impl or 'default'}, "
      f"verification_capacity={verification_capacity}, max_verification_num={max_verification_num}, "
      f"max_draft_len={max_draft_token_length}, max_draft_k={max_draft_k}, "
      f"statistical_time={statistical_time}")
print(f"Draft: train={is_train_draft}")
print(f"Trace: grpo_step_limit={max_grpo_steps}, absolute_target_steps={analysis_enabled}, drift_topk={drift_topk}, "
      f"drift_temperature={drift_temperature}, drift_row_chunk={drift_row_chunk_size}")
print(f"FastGRPO ablation: enabled={fastgrpo_ablation}, draft_lr_multiplier={draft_lr_multiplier}")
print(f"Iteration: grpo_iter={grpo_iteration_num}, sample={sample_num}, "
      f"repeat_gen={repeated_generate_nums}")
print(f"Dataset subset: fraction={train_data_fraction}, max_samples={max_train_samples}, seed={train_subset_seed}")
print("=" * 60)


target_config=AutoConfig.from_pretrained(model_dir)
if model_torch_dtype != "auto":
    target_config.torch_dtype = model_torch_dtype
target_model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=model_torch_dtype, config=target_config, attn_implementation=attn_impl).cuda()
target_model.eval()

config=AutoConfig.from_pretrained(model_dir)
if args.draft_backend == 'eagle3':
    configured_layers = None
    if args.eagle_feature_layers.strip():
        configured_layers = [int(item) for item in args.eagle_feature_layers.split(',')]
    model = Eagle3FastGRPOAdapter(
        target_model=target_model,
        draft_config=args.draft_config,
        draft_checkpoint=adapter_path,
        vocab_mapping=args.vocab_mapping,
        initialization_mode=args.draft_initialization_mode,
        feature_layers=configured_layers,
        ttt_length=args.eagle_ttt_length,
        lk_loss_type=args.eagle_lk_loss_type or None,
        kl_scale=args.eagle_kl_scale,
        kl_decay=args.eagle_kl_decay,
    )
else:
    config.rope_scaling=None
    config.num_hidden_layers=1
    if model_torch_dtype != "auto":
        config.torch_dtype = model_torch_dtype
    model=Model(config,target_model=target_model)
    model.load_model(adapter_path)
print(adapter_path)
model=model.cuda()
tokenizer = AutoTokenizer.from_pretrained(model_dir,padding_side="left")

if args.draft_backend == 'eagle3':
    if len(tokenizer) > int(model.draft_model.vocab_size):
        raise ValueError(
            f'tokenizer size {len(tokenizer)} exceeds EAGLE-3 target vocabulary '
            f'{model.draft_model.vocab_size}'
        )
    if model.draft_model.embed_tokens.weight.shape[0] != model.target_model.get_input_embeddings().weight.shape[0]:
        raise ValueError('EAGLE-3 and target embedding vocabularies are not identical')


if config.model_type == 'llama':
    tokenizer.pad_token = "<|end_of_text|>" 
    tokenizer.pad_token_id = 128001
    

QAs = (
    get_QAs_from_path(args.dataset_path, args.train_split)
    if args.dataset_path else get_train_QAs(args.train_option)
)
full_train_samples = len(QAs)
QAs = select_train_subset(
    QAs,
    fraction=train_data_fraction,
    max_samples=max_train_samples,
    seed=train_subset_seed,
)
selected_train_samples = len(QAs)
print(
    f"Train dataset: option={args.train_option}, full={full_train_samples}, "
    f"selected={selected_train_samples}"
)
df = pd.DataFrame(QAs)

for param in model.draft_model.parameters():
    param.requires_grad=True

for param in model.target_model.parameters():
    param.requires_grad=False
# The EAGLE-3 target-vocabulary head is a view over ``draft_model`` rather
# than an independent target head.  Freezing its parameters would therefore
# freeze the entire draft after it was made trainable above.
if args.draft_backend != 'eagle3':
    for param in model.lm_head.parameters():
        param.requires_grad=False
for param in model.embed_tokens.parameters():
    param.requires_grad=False

draft_trainable_params = sum(
    parameter.numel()
    for parameter in model.draft_model.parameters()
    if parameter.requires_grad
)
if is_train_draft and draft_trainable_params == 0:
    raise RuntimeError('draft training is enabled but the draft has no trainable parameters')
print(f'Draft trainable params: {draft_trainable_params:,}')
    

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,          
    r=64,                           
    lora_alpha=32,                
    lora_dropout=0.0,              
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

model.target_model = get_peft_model(model.target_model,lora_config)
if  args.load_lora_path != "":
    model.target_model.load_adapter(args.load_lora_path,adapter_name="default")
model.target_model.print_trainable_parameters()

def _get_base_causal_lm(causal_lm):
    """Return the underlying causal LM while preserving injected LoRA modules."""
    if hasattr(causal_lm, "get_base_model"):
        return causal_lm.get_base_model()
    if hasattr(causal_lm, "base_model") and hasattr(causal_lm.base_model, "model"):
        return causal_lm.base_model.model
    return causal_lm


def _autocast_dtype(causal_lm):
    dtype = getattr(causal_lm, "dtype", None)
    if dtype == torch.bfloat16:
        return torch.bfloat16
    return torch.float16


def _token_logps_from_hidden(hidden_states, lm_head, labels, chunk_size):
    """Compute selected-token log-probabilities without a full [B, T, vocab] tensor."""
    hidden_states = hidden_states[:, :-1, :]
    labels = labels[:, 1:].to(hidden_states.device)
    seq_len = hidden_states.shape[1]
    logps_chunks = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = lm_head(hidden_states[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        selected_logits = torch.gather(
            logits, dim=-1, index=cur_labels.unsqueeze(-1)
        ).squeeze(-1)
        log_denominator = torch.logsumexp(logits, dim=-1)
        logps_chunks.append(selected_logits - log_denominator)
        del logits, selected_logits, log_denominator

    if logps_chunks:
        return torch.cat(logps_chunks, dim=1)
    return hidden_states.new_zeros((hidden_states.shape[0], 0))


def compute_model_token_logps(causal_lm, input_ids, attention_mask, chunk_size):
    """Forward the backbone once, then apply the LM head in bounded chunks."""
    base_model = _get_base_causal_lm(causal_lm)
    device = input_ids.device
    device_type = "cuda" if device.type == "cuda" else device.type

    with torch.amp.autocast(
        device_type,
        dtype=_autocast_dtype(base_model),
        enabled=(device.type == "cuda"),
    ):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

    return _token_logps_from_hidden(
        hidden_states, base_model.lm_head, input_ids, chunk_size
    )


def compute_target_loss_and_backward(
    model,
    input_ids,
    attention_mask,
    mask,
    reward,
    epsilon,
    beta,
    grpo_iteration,
    old_logps=None,
    ref_logps=None,
    chunk_size=256,
    loss_scale=1.0,
):
    """Compute the GRPO loss and backpropagate without full-vocabulary logits."""
    device = input_ids.device
    token_mask = mask[:, :-1].to(device=device, dtype=torch.float32)
    denom = token_mask.sum(-1).clamp_min(1.0)
    reward = reward.to(device=device, dtype=torch.float32)
    seq_len = token_mask.shape[1]

    if grpo_iteration == 0:
        model.target_model.disable_adapter_layers()
        with torch.no_grad():
            ref_logps_gpu = compute_model_token_logps(
                model.target_model,
                input_ids,
                attention_mask,
                chunk_size,
            ).detach()
        model.target_model.enable_adapter_layers()
        ref_logps_for_loss = ref_logps_gpu
        old_logps_for_loss = None
    else:
        if old_logps is None or ref_logps is None:
            raise ValueError(
                "old_logps and ref_logps are required when grpo_iteration > 0"
            )
        old_logps_for_loss = old_logps.to(device, non_blocking=True)
        ref_logps_for_loss = ref_logps.to(device, non_blocking=True)

    model.target_model.enable_adapter_layers()
    base_model = _get_base_causal_lm(model.target_model)
    device_type = "cuda" if device.type == "cuda" else device.type
    with torch.amp.autocast(
        device_type,
        dtype=_autocast_dtype(base_model),
        enabled=(device.type == "cuda"),
    ):
        outputs = base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )

    policy_hidden = hidden_states[:, :-1, :]
    labels = input_ids[:, 1:].to(device)
    old_chunks_to_store = []
    loss_value = 0.0
    abs_loss1_value = 0.0
    loss2_value = 0.0

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        logits = base_model.lm_head(policy_hidden[:, start:end, :]).float()
        cur_labels = labels[:, start:end]
        logps = torch.gather(
            logits, dim=-1, index=cur_labels.unsqueeze(-1)
        ).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        cur_mask = token_mask[:, start:end]

        if grpo_iteration == 0:
            cur_old_logps = logps.detach()
            old_chunks_to_store.append(cur_old_logps.cpu())
        else:
            cur_old_logps = old_logps_for_loss[:, start:end]

        cur_ref_logps = ref_logps_for_loss[:, start:end]
        coef1 = torch.exp(logps - cur_old_logps)
        coef2 = torch.clamp(coef1, 1 - epsilon, 1 + epsilon)
        loss1 = torch.min(coef1 * reward, coef2 * reward)

        coef3 = cur_ref_logps - logps
        kl = torch.exp(coef3) - coef3 - 1
        token_loss = -(loss1 - beta * kl)
        chunk_loss = ((token_loss * cur_mask).sum(-1) / denom).sum()
        scaled_loss = chunk_loss * loss_scale
        scaled_loss.backward(retain_graph=(end < seq_len))

        with torch.no_grad():
            loss_value += float(chunk_loss.detach().cpu())
            abs_loss1_value += float(
                torch.abs((loss1 * cur_mask).sum(-1) / denom).sum().detach().cpu()
            )
            loss2_value += float(
                ((kl * cur_mask).sum(-1) / denom).sum().detach().cpu()
            )

        del (
            logits,
            logps,
            coef1,
            coef2,
            loss1,
            coef3,
            kl,
            token_loss,
            chunk_loss,
            scaled_loss,
        )

    if grpo_iteration == 0:
        if old_chunks_to_store:
            old_logps_out = torch.cat(old_chunks_to_store, dim=1)
        else:
            old_logps_out = torch.empty((input_ids.shape[0], 0))
        ref_logps_out = ref_logps_for_loss.detach().cpu()
    else:
        old_logps_out = old_logps
        ref_logps_out = ref_logps

    del hidden_states, policy_hidden
    if grpo_iteration == 0:
        del ref_logps_gpu, ref_logps_for_loss
    else:
        del old_logps_for_loss, ref_logps_for_loss

    return (
        loss_value,
        abs_loss1_value,
        loss2_value,
        old_logps_out,
        ref_logps_out,
    )


def training_draft_model(model,outputs,prompt_mask,token_budget=None):
    if getattr(model, 'is_eagle3_specforge', False):
        return training_eagle3_specforge(model, outputs, prompt_mask, token_budget=token_budget)
    

    all_draft_input_states = outputs['all_draft_input_states']
    all_draft_input_ids = outputs['all_draft_input_ids']
    all_prompt_length = [prompt_mask[idx // repeated_generate_nums].sum().item() for idx in range(len(all_draft_input_states))]
    
    prompt_mask=prompt_mask.cpu()
    device=model.target_model.device
    
    sorted_pairs = sorted(
        zip(all_draft_input_ids, all_draft_input_states, all_prompt_length),
        key=lambda x: len(x[0]),
        reverse=False  
    )

    all_draft_input_ids_sorted, all_draft_input_states_sorted, all_prompt_length_sorted = zip(*sorted_pairs)

    all_draft_input_ids = list(all_draft_input_ids_sorted)
    all_draft_input_states = list(all_draft_input_states_sorted)
    all_prompt_length = list(all_prompt_length_sorted)
    
    l1_loss=torch.nn.SmoothL1Loss(reduction='none')
    total_loss1,total_loss2=0,0
    drift_tv_sum = 0.0
    drift_kl_sum = 0.0
    drift_count = 0
    
    draft_input_states_list=[]
    draft_input_ids_list=[]
    prompt_length_list=[]
    
    cur_max_length=0
    hidden_size=all_draft_input_states[0].shape[-1]
    
    for idx , (draft_input_states,draft_input_ids,prompt_length) in enumerate(zip(all_draft_input_states,all_draft_input_ids,all_prompt_length)):
        
        if ((draft_input_ids.shape[-1]*(len(draft_input_states_list)+1)<=max_training_token*2 and
            (draft_input_ids.shape[-1]-cur_max_length)*len(draft_input_states_list)<=max_training_padding_gap) or
            len(draft_input_states_list)==0):
            
                draft_input_states_list.append(draft_input_states)
                draft_input_ids_list.append(draft_input_ids)
                prompt_length_list.append(prompt_length)
                
                cur_max_length=max(cur_max_length, draft_input_ids.shape[-1])
            
        else:
            
            cur_batch=len(draft_input_states_list)

            loss_mask=[[] for _ in range(cur_batch)]
            attention_mask=[[] for _ in range(cur_batch)]
            
            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
                attention_mask[idx_seq]=[1]*cur_len

            for idx_seq in range(cur_batch):
                cur_len=draft_input_ids_list[idx_seq].shape[-1]
                padding_len=cur_max_length-cur_len
                
                if padding_len>0:
                    draft_input_states_list[idx_seq]=torch.concat(
                        [draft_input_states_list[idx_seq],
                        torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                        dim=-2)
                    
                    draft_input_ids_list[idx_seq]=torch.concat(
                        [draft_input_ids_list[idx_seq],
                        torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                        dim=-1)
                    
                    loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
                    attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
            
            draft_input_states=torch.stack(draft_input_states_list,dim=0)
            draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
            loss_mask=torch.tensor(loss_mask,device=device)
            attention_mask=torch.tensor(attention_mask,device=device)

            with torch.amp.autocast(str(model.target_model.device),
                        dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
                draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                                attention_mask=attention_mask,use_cache=False)
                
            next_feature_states=draft_outputs['next_feature_states']
            draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
            draft_logits=model.lm_head(draft_hidden_states)
            
            with torch.no_grad():
                target_hidden_states=draft_input_states
                target_logits_raw=model.target_model.lm_head(target_hidden_states.to(model.target_model.dtype))[:,1:,:]

            draft_logits_raw=draft_logits[:,:-1,:]
            drift_tv, drift_kl, cur_drift_count = acceptance_aligned_union_metrics(
                draft_logits_raw.detach(),
                target_logits_raw.detach(),
                loss_mask[...,:-1].bool(),
                topk=drift_topk,
                temperature=drift_temperature,
                row_chunk_size=drift_row_chunk_size,
            )
            drift_tv_sum += drift_tv * cur_drift_count
            drift_kl_sum += drift_kl * cur_drift_count
            drift_count += cur_drift_count
            target_logits=target_logits_raw.float().softmax(dim=-1).detach()
                
            loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

            loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
            loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
            loss1=loss1.sum(-1)
            loss1=loss1*2.0
            
            draft_logits=draft_logits_raw.float().softmax(dim=-1)

            plogp=target_logits*torch.log(draft_logits)
            loss2=torch.sum(plogp,dim=-1)*loss_mask[...,:-1]
            loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
            loss2= - loss2.sum(-1)

            loss2=loss2*0.1
            
            loss=loss1+loss2
                
            total_loss1+=loss1.item()
            total_loss2+=loss2.item()
            
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                
                loss = loss.detach()
                del loss
                torch.cuda.empty_cache()
            else:

                loss=loss/len(all_draft_input_states)
                loss=loss/draft_accumulation_steps
                loss.backward()
                
            draft_input_states_list=[all_draft_input_states[idx]]
            draft_input_ids_list=[all_draft_input_ids[idx]]
            prompt_length_list=[all_prompt_length[idx]]
            cur_max_length=all_draft_input_ids[idx].shape[-1]
            
    cur_batch=len(draft_input_states_list)

    loss_mask=[[] for _ in range(cur_batch)]
    attention_mask=[[] for _ in range(cur_batch)]
    
    cur_max_length=0
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        loss_mask[idx_seq]=[0]*prompt_length_list[idx_seq]+[1]*(cur_len-prompt_length_list[idx_seq])
        attention_mask[idx_seq]=[1]*cur_len
        
        cur_max_length=max(cur_max_length, cur_len)
        
    for idx_seq in range(cur_batch):
        cur_len=draft_input_ids_list[idx_seq].shape[-1]
        padding_len=cur_max_length-cur_len
        
        if padding_len>0:
            draft_input_states_list[idx_seq]=torch.concat(
                [draft_input_states_list[idx_seq],
                torch.zeros((padding_len, hidden_size), dtype=draft_input_states_list[idx_seq].dtype, device=device)],
                dim=-2)
            
            draft_input_ids_list[idx_seq]=torch.concat(
                [draft_input_ids_list[idx_seq],
                torch.zeros(padding_len, dtype=draft_input_ids_list[idx_seq].dtype, device=device)],
                dim=-1)
            
            loss_mask[idx_seq]=loss_mask[idx_seq]+[0]*padding_len
            attention_mask[idx_seq]=attention_mask[idx_seq]+[0]*padding_len
    
    draft_input_states=torch.stack(draft_input_states_list,dim=0)
    draft_input_ids=torch.stack(draft_input_ids_list,dim=0)
    loss_mask=torch.tensor(loss_mask,device=device)
    attention_mask=torch.tensor(attention_mask,device=device)
    
    with torch.amp.autocast(str(model.target_model.device),
                dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        draft_outputs=model(hidden_states=draft_input_states,input_ids=draft_input_ids,
                        attention_mask=attention_mask,use_cache=False)
        
    next_feature_states=draft_outputs['next_feature_states']
    draft_hidden_states=draft_outputs['hidden_states'].to(model.target_model.dtype)
    draft_logits=model.lm_head(draft_hidden_states)
    
    with torch.no_grad():
        target_hidden_states=draft_input_states
        target_logits_raw=model.target_model.lm_head(target_hidden_states.to(model.target_model.dtype))[:,1:,:]

    draft_logits_raw=draft_logits[:,:-1,:]
    drift_tv, drift_kl, cur_drift_count = acceptance_aligned_union_metrics(
        draft_logits_raw.detach(),
        target_logits_raw.detach(),
        loss_mask[...,:-1].bool(),
        topk=drift_topk,
        temperature=drift_temperature,
        row_chunk_size=drift_row_chunk_size,
    )
    drift_tv_sum += drift_tv * cur_drift_count
    drift_kl_sum += drift_kl * cur_drift_count
    drift_count += cur_drift_count
    target_logits=target_logits_raw.float().softmax(dim=-1).detach()
        
    loss1=l1_loss(next_feature_states[:,:-1,:].float(),draft_input_states[:,1:,:].float())

    loss1=torch.mean(loss1,dim=-1)*loss_mask[...,:-1] 
    loss1=torch.sum(loss1, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
    loss1=loss1.sum(-1)
    loss1=loss1*2.0
    
    draft_logits=draft_logits_raw.float().softmax(dim=-1)

    plogp=target_logits*torch.log(draft_logits)
    loss2=torch.sum(plogp,dim=-1)*loss_mask[...,:-1]
    loss2=torch.sum(loss2, dim=-1) / torch.sum(loss_mask[...,:-1], dim=-1)
    loss2= - loss2.sum(-1)

    loss2=loss2*0.1
    
    loss=loss1+loss2
        
    total_loss1+=loss1.item()
    total_loss2+=loss2.item()
    
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        
        loss = loss.detach()
        del loss
        torch.cuda.empty_cache()
    else:

        loss=loss/len(all_draft_input_states)
        loss.backward()
            
        
    total_loss1/=len(all_draft_input_states)
    total_loss2/=len(all_draft_input_states)
    
    mean_drift_tv = drift_tv_sum / max(drift_count, 1)
    mean_drift_kl = drift_kl_sum / max(drift_count, 1)
    return total_loss1,total_loss2,mean_drift_tv,mean_drift_kl,drift_count


def training_eagle3_specforge(model, outputs, prompt_mask, token_budget=None):
    """Use SpecForge's EAGLE-3 architecture, KL/LK loss and TTT unrolling.

    This adapter only packs FastGRPO rollout tensors.  Objective computation and
    recursive unrolling are performed by ``OnlineEagle3Model.forward``.
    """
    feature_rows = outputs['all_draft_input_states']
    target_rows = outputs.get('all_target_hidden_states')
    token_rows = outputs['all_draft_input_ids']
    if target_rows is None:
        raise RuntimeError('EAGLE-3 rollout is missing final target hidden states')
    if not (len(feature_rows) == len(target_rows) == len(token_rows)):
        raise RuntimeError('misaligned EAGLE-3 rollout feature tensors')
    total_loss = 0.0
    total_acceptance = 0.0
    valid_tokens = 0
    remaining_budget = None if token_budget is None else int(token_budget)
    training_model = model.specforge_training_model
    target_head = _get_base_causal_lm(model.target_model).lm_head.weight
    for index, (features, target_hidden, input_ids_row) in enumerate(
        zip(feature_rows, target_rows, token_rows)
    ):
        # Rollouts are intentionally collected under inference_mode. Clone at
        # the training boundary so autograd may save these tensors for the
        # draft-weight backward pass.
        features = features.detach().clone()
        target_hidden = target_hidden.detach().clone()
        input_ids_row = input_ids_row.detach().clone()
        prompt_len = int(prompt_mask[index // repeated_generate_nums].sum().item())
        input_ids_row, features, target_hidden, row_loss_mask = aligned_eagle3_row(
            features, target_hidden, input_ids_row, prompt_len, remaining_budget
        )
        seq_len = int(input_ids_row.shape[-1])
        loss_mask = row_loss_mask.unsqueeze(0)
        cur_valid = int(loss_mask.sum().item())
        if cur_valid == 0:
            continue
        result = training_model(
            input_ids=input_ids_row.unsqueeze(0),
            attention_mask=torch.ones((1, seq_len), device=model.device, dtype=torch.long),
            target=None,
            loss_mask=loss_mask,
            hidden_states=features.unsqueeze(0).to(model.dtype),
            target_hidden_for_compact=target_hidden.unsqueeze(0).to(model.dtype),
            target_head_weight=target_head,
        )
        plosses, acceptance_rates = result[0], result[1]
        loss = torch.stack([item.float() for item in plosses]).sum()
        if not loss.requires_grad:
            raise RuntimeError(
                'EAGLE-3 loss has no gradient path to the draft parameters; '
                f'draft_trainable_params={draft_trainable_params}'
            )
        (loss / max(len(feature_rows), 1)).backward()
        total_loss += float(loss.detach().cpu())
        total_acceptance += sum(float(item.detach().cpu()) for item in acceptance_rates)
        valid_tokens += cur_valid
        if remaining_budget is not None:
            remaining_budget -= cur_valid
            if remaining_budget <= 0:
                break
    denom = max(len(feature_rows), 1)
    # Keep the legacy return arity. The second field is the SpecForge simulated
    # acceptance objective; sparse legacy drift metrics are deliberately absent.
    return total_loss / denom, total_acceptance / denom, 0.0, 0.0, valid_tokens

        
optimizer_target = torch.optim.AdamW(model.target_model.parameters(), lr=target_lr)
optimizer_draft = torch.optim.AdamW(model.draft_model.parameters(), lr=draft_lr)

log_mode = "a" if append_log else "w"
with open(log_file, log_mode, encoding='utf-8') as f:
    pass

step=0
target_optimizer_steps=0
used_items=0
draft_step=0
draft_accumulated_step=0 
batch_logs=[]
batch_data={
    'messages':[],
    'rewards':[],
    'std_rewards':[],
    'response_ids':[],
    'generate_time_cost':0,
    'last_generate_time_cost':[],
    'train_time_cost':0,
    'last_train_time_cost':[],
    'generate_length':0,
    'last_generate_length':[],
    'total_rollout_tokens':0,
    'total_acc_length':0,
    'last_acc_length':[],
    'total_decoded_token_num':0,
    'last_decoded_token_num':[],
    'total_accepted_draft_tokens':0,
    'total_proposed_draft_tokens':0,
    'last_accepted_draft_tokens':[],
    'last_proposed_draft_tokens':[],
    'prefill_time_cost':0,
    'target_time_cost':0,
    'draft_time_cost':0,
    'check_time_cost':0,
    'ignore_due_correct':0,
    'ignore_due_incorrect':0,
    'mean_rewards':0,
    'last_mean_rewards':[],
    'draft_train_time_cost':0,
    'last_draft_loss1':[],
    'last_draft_loss2':[] ,
    'generate_length_list':[],
    'draft_sparse_tv_sum':0.0,
    'draft_sparse_kl_sum':0.0,
    'draft_sparse_count':0,
    'trace_rollout_count':0,
}

optimizer_target.zero_grad(set_to_none=True)
optimizer_draft.zero_grad(set_to_none=True)
start_time=time.time()
batch=[]
start_epoch = 0
start_batch = 0
last_checkpoint_step = -1

if resume_checkpoint:
    checkpoint = load_training_checkpoint(
        resume_checkpoint,
        model=model,
        optimizer_target=optimizer_target,
        optimizer_draft=optimizer_draft,
    )
    step = int(checkpoint.get("step", 0))
    if checkpoint.get("target_optimizer_steps") is not None:
        target_optimizer_steps = int(checkpoint["target_optimizer_steps"])
    else:
        # Older checkpoints did not store this counter. AdamW's parameter
        # states still record the number of committed target updates.
        target_optimizer_steps = max(
            (int(state['step']) for state in optimizer_target.state.values() if 'step' in state),
            default=0,
        )
    used_items = int(checkpoint.get("used_items", 0))
    draft_step = int(checkpoint.get("draft_step", 0))
    draft_accumulated_step = int(checkpoint.get("draft_accumulated_step", 0))
    saved_batch_data = checkpoint.get("batch_data", {})
    if isinstance(saved_batch_data, dict):
        batch_data.update(saved_batch_data)
    start_epoch = int(checkpoint.get("epoch", 0))
    start_batch = int(checkpoint.get("next_batch", 0))
    last_checkpoint_step = target_optimizer_steps if analysis_enabled else step
    print(
        f"Resumed FastGRPO checkpoint {resume_checkpoint}: "
        f"epoch={start_epoch + 1}, next_batch={start_batch}, "
        f"step={step}, used_items={used_items}, draft_step={draft_step}"
    )
    if reset_rng_on_resume:
        _seed_everything(trace_seed)
        print(f"Reset continuation RNG state to paired trace seed={trace_seed}")

# A continuation trace uses local counters so --max_grpo_steps=100 means 100
# newly completed labels even when the restored checkpoint is already at 310.
trace_start_step = int(step)
trace_start_target_optimizer_steps = int(target_optimizer_steps)
trace_start_draft_step = int(draft_step)
trace_rollout_count = 0
stop_requested = bool(analysis_enabled and max_grpo_steps > 0
                      and target_optimizer_steps >= max_grpo_steps)
batch_data['draft_sparse_tv_sum'] = 0.0
batch_data['draft_sparse_kl_sum'] = 0.0
batch_data['draft_sparse_count'] = 0
batch_data['trace_rollout_count'] = 0
for param_group in optimizer_draft.param_groups:
    # AdamW.load_state_dict restores the checkpoint LR. The requested run LR
    # must win, otherwise old checkpoints silently change both branch updates.
    param_group['lr'] = float(draft_lr) * draft_lr_multiplier
effective_draft_lrs = [float(group['lr']) for group in optimizer_draft.param_groups]
if analysis_enabled and (len(set(effective_draft_lrs)) != 1 or
                         effective_draft_lrs[0] != float(draft_lr)):
    raise RuntimeError('policy-lag branch LR must equal the requested --draft_lr')

analysis_dependencies = {}
if analysis_enabled:
    dependency_path = analysis_root / 'dependencies.json'
    if dependency_path.is_file():
        analysis_dependencies = json.loads(dependency_path.read_text(encoding='utf-8'))

run_config_log = {
    "phase": "run_config",
    "run_name": version_name,
    "resume_checkpoint": str(resume_checkpoint),
    "append_log": bool(append_log),
    "source_grpo_step": int(trace_start_step),
    "source_target_optimizer_steps": int(trace_start_target_optimizer_steps),
    "source_draft_step": int(trace_start_draft_step),
    "max_grpo_steps": int(max_grpo_steps),
    "num_epochs": int(num_epochs),
    "train_batch_size": int(batch_size),
    "responses_per_prompt": int(repeated_generate_nums),
    "target_lr_requested": float(target_lr),
    "draft_lr_requested": float(draft_lr),
    "analysis_boundaries": sorted(analysis_boundaries),
    "analysis_interval": int(analysis_interval),
    "train_option": str(args.train_option),
    "dataset_path": str(args.dataset_path),
    "eval_dataset_path": str(args.eval_dataset_path),
    "train_data_fraction": float(train_data_fraction),
    "train_subset_seed": int(train_subset_seed),
    "seed": int(trace_seed),
    "reset_rng_on_resume": bool(reset_rng_on_resume),
    "drift_metric": "bidirectional_topk_union_with_tail_sparse_tv",
    "drift_kl_direction": "forward_kl_target_to_draft",
    "drift_topk": int(drift_topk),
    "drift_temperature": float(drift_temperature),
    "drift_normalization": "mean_over_valid_response_token_rows",
    "draft_lr_multiplier": float(draft_lr_multiplier),
    "effective_draft_lrs": effective_draft_lrs,
    "analysis_protocol": PROTOCOL_VERSION if analysis_enabled else None,
    "reflex_config": reflex_config if analysis_enabled else None,
    "analysis_keep_branch_checkpoints": analysis_keep_branch_checkpoints,
    "analysis_eval_prompts": analysis_eval_prompts if analysis_enabled else None,
    "analysis_eval_batch_size": analysis_eval_batch_size if analysis_enabled else None,
    "analysis_requested_training_token_budget": int(args.analysis_training_token_budget),
    "analysis_draft_update_steps": int(args.analysis_draft_update_steps),
    "target_max_training_token": int(max_training_token),
    "train_eval_disjointness": analysis_dependencies.get('train_eval_disjointness'),
    "source_revision": analysis_dependencies.get('fastgrpo_commit'),
    "source_sha256": source_sha256 if analysis_enabled else None,
    "fastgrpo_ablation": bool(fastgrpo_ablation),
}
with open(log_file, 'a', encoding='utf-8') as f:
    f.write(json.dumps(run_config_log) + '\n')
print(json.dumps(run_config_log, sort_keys=True))

class TrainDataCollator:
    def __init__(self, tokenizer, max_prompt_length):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
    
    def __call__(self, batch):
        system_prompt = "You are a math problem assistant." 
        user_prompt =  '''Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}'''
        messages = []
        answers = []

        for example in batch:
            messages.append([
                {"role" : "system" , "content": system_prompt} , 
                {"role" : "user" , "content": user_prompt.format_map({"instruction" : example['question']}) }
            ])
            answers.append(example['answer'])
        tokenized_inputs = self.tokenizer(
            text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
            return_tensors='pt',padding='longest',truncation=True,max_length=self.max_prompt_length,padding_side='left'
        )

        return {
            'input_ids': tokenized_inputs['input_ids'],
            'attention_mask': tokenized_inputs['attention_mask'],
            'messages': messages,        
            'answers': answers,           
        }


def _analysis_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_analysis_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _analysis_boundary(step_value):
    return is_analysis_boundary(step_value, analysis_boundaries, analysis_interval, max_grpo_steps)


def _teacher_prefix_hidden(eval_batch):
    base = _get_base_causal_lm(model.target_model)
    rows = []
    for start in range(0, len(eval_batch['input_ids']), analysis_eval_batch_size):
        stop = start + analysis_eval_batch_size
        with torch.inference_mode():
            result = base.model(
                input_ids=eval_batch['input_ids'][start:stop].to(model.target_model.device),
                attention_mask=eval_batch['attention_mask'][start:stop].to(model.target_model.device),
                use_cache=False,
                return_dict=True,
            )
            hidden = result.last_hidden_state if hasattr(result, 'last_hidden_state') else result[0]
            rows.append(hidden.float().cpu())
    return torch.cat(rows, dim=0)


def _teacher_tv_from_hidden(old_hidden, new_hidden, valid_mask):
    base = _get_base_causal_lm(model.target_model)
    total = 0.0
    count = 0
    for batch_start in range(0, old_hidden.shape[0], analysis_eval_batch_size):
        batch_stop = batch_start + analysis_eval_batch_size
        for start in range(0, old_hidden.shape[1], drift_row_chunk_size):
            stop = start + drift_row_chunk_size
            mask = valid_mask[batch_start:batch_stop, start:stop].bool()
            if not mask.any():
                continue
            with torch.inference_mode():
                old_logits = base.lm_head(
                    old_hidden[batch_start:batch_stop, start:stop].to(
                        device=model.target_model.device, dtype=base.lm_head.weight.dtype
                    )
                )
                new_logits = base.lm_head(
                    new_hidden[batch_start:batch_stop, start:stop].to(
                        device=model.target_model.device, dtype=base.lm_head.weight.dtype
                    )
                )
                value = teacher_shift_tv(old_logits, new_logits, mask, row_chunk_size=drift_row_chunk_size)
            cur_count = int(mask.sum().item())
            total += value * cur_count
            count += cur_count
            del old_logits, new_logits
    return total / max(count, 1)


def _analysis_rows_from_rollout(branch_name, result, policy_step, epoch, batch_index, *, used_for_grpo):
    rows = []
    lengths = tuple(len(result[key]) for key in (
        'response_accepted_length_sum', 'response_verification_rounds', 'response_generated_tokens'
    ))
    if len(set(lengths)) != 1 or lengths[0] % repeated_generate_nums != 0:
        raise RuntimeError(f'{branch_name} rollout response metrics are misaligned: {lengths}')
    for response_index, (accepted, rounds, generated) in enumerate(zip(
        result['response_accepted_length_sum'],
        result['response_verification_rounds'],
        result['response_generated_tokens'],
    )):
        rows.append({
            'policy_step': int(policy_step),
            'seed': -1,  # The real training RNG is captured, not replaced by a fixed seed.
            'rng_source': 'captured_training_rng',
            'branch': branch_name,
            'evaluation_scope': 'next_real_grpo_rollout',
            'evaluation_epoch': int(epoch + 1),
            'evaluation_batch': int(batch_index),
            'used_for_grpo': bool(used_for_grpo),
            'prompt_id': int(response_index // repeated_generate_nums),
            'response_index': int(response_index % repeated_generate_nums),
            'accepted_length_sum': int(accepted),
            'verification_rounds': int(rounds),
            'generated_tokens': int(generated),
        })
    if sum(row['accepted_length_sum'] for row in rows) != int(result['total_acc_length']):
        raise RuntimeError(f'{branch_name} per-response accepted lengths do not match FastGRPO total')
    if sum(row['verification_rounds'] for row in rows) != int(result['total_decoded_token_num']):
        raise RuntimeError(f'{branch_name} per-response verification rounds do not match FastGRPO total')
    return rows


def _shadow_persistent_identity():
    # Version/storage checks include frozen base weights without copying 3B
    # parameters to CPU for every shadow. Hash trainable LoRA, optimizer and
    # the real response buffer as well.
    return (state_digest(_target_lora_state_dict(model.target_model)),
            tuple((name, parameter._version, parameter.data_ptr())
                  for name, parameter in model.target_model.named_parameters()),
            state_digest(optimizer_draft.state_dict()), state_digest(batch_data))


def _evaluate_analysis_branch(branch_name, draft_state, stale_state, eval_batch, pending,
                              epoch, batch_index, rng_state):
    policy_step = int(pending['policy_step'])
    state_branch, _ = branch_spec(branch_name)
    eval_prompt_batch_id = state_digest(eval_batch)
    modes = [(module, module.training) for module in model.modules()]

    def rollout(reflex_mode):
        model.eval()
        with torch.inference_mode():
            return speculative_generate_in_prompt_batches(
                prompt_batch_size=analysis_eval_batch_size,
                model=model,
                input_ids=eval_batch['input_ids'].to('cuda'),
                attention_mask=eval_batch['attention_mask'].to('cuda'),
                tokenizer=tokenizer,
                do_sample=True,
                max_length=max_length,
                repeated_generate_nums=repeated_generate_nums,
                temperature=temperature,
                top_p=top_p,
                verification_capacity=verification_capacity,
                max_draft_token_length=max_draft_token_length,
                max_draft_k=max_draft_k,
                max_verification_num=max_verification_num,
                min_draft_token_length=min_draft_token_length,
                draft_token_length_c=draft_token_length_c,
                return_all_draft_input=False,
                statistical_time=False,
                reflex_mode=reflex_mode,
                reflex_time_updates=reflex_mode == 'active',
                **reflex_config,
            )
    try:
        result = paired_shadow_rollout(
            branch=branch_name, branch_state=draft_state, stale_state=stale_state,
            rng_state=rng_state, expected_draft_id=pending[f'{state_branch}_draft_id'],
            digest=state_digest, load_draft=lambda state: model.draft_model.load_state_dict(state, strict=True),
            read_draft=lambda: model.draft_model.state_dict(), persistent_identity=_shadow_persistent_identity,
            capture_rng=_analysis_rng_state, restore_rng=_restore_analysis_rng, rollout=rollout,
        )
    finally:
        for module, mode in modes:
            module.training = mode
    if branch_name == 'reflex' and not (
        result['reflex_state_initialized_zero'] and result['reflex_state_cleared']
    ):
        raise RuntimeError('Reflex fast state leaked across rollouts')
    if state_digest(eval_batch) != eval_prompt_batch_id:
        raise RuntimeError(f'{branch_name} shadow modified the paired prompt batch')
    rows = _analysis_rows_from_rollout(branch_name, result, policy_step, epoch, batch_index, used_for_grpo=False)
    for row in rows:
        row['initial_rng_id'] = state_digest(rng_state)
        row['evaluation_prompt_batch_id'] = eval_prompt_batch_id
    return rows, {key: result.get(key) for key in (
        'reflex_update_gpu_ms', 'reflex_update_count', 'reflex_update_gpu_ms_per_update',
        'reflex_update_exposed_ms')}


def _finalize_next_rollout_analysis(pending, stale_rows, fresh_rows, reflex_rows, reflex_timing, epoch, batch_index,
                                    next_target_optimizer_step, eval_prompt_batch_id):
    policy_step = int(pending['policy_step'])
    if next_target_optimizer_step != policy_step + 1:
        raise RuntimeError('live rollout was not followed by exactly one target optimizer update')
    expected_responses = analysis_eval_prompts * repeated_generate_nums
    if any(len(rows) != expected_responses for rows in (stale_rows, fresh_rows, reflex_rows)):
        raise RuntimeError('analysis branches did not evaluate all requested real-rollout responses')
    if len({row['initial_rng_id'] for rows in (stale_rows, fresh_rows, reflex_rows) for row in rows}) != 1:
        raise RuntimeError('analysis branches used different initial RNG states')
    if any(row['evaluation_prompt_batch_id'] != eval_prompt_batch_id
           for rows in (stale_rows, fresh_rows, reflex_rows) for row in rows):
        raise RuntimeError('analysis branches used different prompt batches')
    boot = {branch: bootstrap_delta_by_prompt(
        stale_rows, records, seed=trace_seed + policy_step,
        samples=int(args.analysis_bootstrap_samples))
        for branch, records in (('fresh', fresh_rows), ('reflex', reflex_rows))}
    analysis_per_response[:] = [
        row for row in analysis_per_response if int(row['policy_step']) != policy_step
    ]
    analysis_summaries[:] = [
        row for row in analysis_summaries if int(row.policy_step) != policy_step
    ]
    boundary_responses = stale_rows + fresh_rows + reflex_rows
    for row in boundary_responses:
        row['evaluation_prompt_batch_id'] = eval_prompt_batch_id
    analysis_per_response.extend(boundary_responses)
    boundary_summaries = []
    for branch_name, records in (('stale', stale_rows), ('fresh', fresh_rows), ('reflex', reflex_rows)):
        aal, accepted, rounds, generated = weighted_aal(records)
        state_branch, _ = branch_spec(branch_name)
        delta = boot.get(branch_name, {})
        boundary_summaries.append(BranchSummary(
            policy_step=policy_step,
            seed=-1,
            branch=branch_name,
            aal=aal,
            delta_aal=delta.get('delta_aal'),
            verification_rounds=rounds,
            generated_tokens=generated,
            actual_training_token_count=int(pending['common_training_token_budget']),
            optimizer_steps=int(pending['optimizer_steps_per_branch']),
            policy_checkpoint_id=pending['policy_checkpoint_id'],
            draft_checkpoint_id=pending[f'{state_branch}_draft_id'],
            feature_policy_version=(
                pending['old_policy_id'] if state_branch == 'stale'
                else pending['policy_checkpoint_id']
            ),
            teacher_shift_tv=float(pending['teacher_shift_tv']),
            ci_low=delta.get('delta_aal_ci_low'),
            ci_high=delta.get('delta_aal_ci_high'),
            evaluation_epoch=int(epoch + 1),
            evaluation_batch=int(batch_index),
            used_for_grpo=branch_name == 'stale',
            evaluation_prompt_batch_id=eval_prompt_batch_id,
            requested_training_token_budget=int(pending['requested_training_token_budget']),
            effective_draft_lr=float(pending['effective_draft_lr']),
            accepted_length_sum=accepted,
            online_draft_update_gpu_ms=pending[f'{state_branch}_online_draft_update_gpu_ms'],
            analysis_io_wall_ms=pending['analysis_io_wall_ms'],
            **(reflex_timing if branch_name == 'reflex' else {}),
        ))
    analysis_summaries.extend(boundary_summaries)
    completion = {
        **pending,
        'evaluation_epoch': int(epoch + 1),
        'evaluation_batch': int(batch_index),
        'evaluation_prompt_count': len({row['prompt_id'] for row in stale_rows}),
        'evaluation_prompt_batch_id': eval_prompt_batch_id,
        'next_target_optimizer_step': int(next_target_optimizer_step),
        'stale_rollout_used_for_grpo': True,
        'main_branch': 'stale',
        'durable_target_checkpoint': False,
    }
    metrics = append_boundary_results(Path(policy_lag_output_dir), boundary_responses, boundary_summaries,
                                      completion, io_meter=analysis_io)
    with analysis_io.measure():
        with open(log_file, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps({'phase': 'analysis_boundary', **metrics}) + '\n')
    print(json.dumps({'phase': 'analysis_boundary', **metrics}, sort_keys=True))
    analysis_completed.add(policy_step)


if analysis_enabled:
    Path(policy_lag_output_dir).mkdir(parents=True, exist_ok=True)
pending_analysis = None
if analysis_enabled:
    pending_paths = []
    for pending_path in Path(policy_lag_output_dir).glob('boundaries/step_*/pending.json'):
        complete_path = pending_path.parent / 'complete.json'
        if complete_path.is_file():
            completion = json.loads(complete_path.read_text(encoding='utf-8'))
            # Results become durable only with the *following* target
            # checkpoint. If we resumed an older checkpoint, replay the live
            # rollout and replace the uncommitted result rows.
            if int(completion.get('next_target_optimizer_step', 0)) <= target_optimizer_steps:
                continue
            analysis_completed.discard(int(completion['policy_step']))
        metadata = json.loads(pending_path.read_text(encoding='utf-8'))
        pending_step = int(metadata['policy_step'])
        if pending_step < target_optimizer_steps:
            raise RuntimeError(
                f'unfinished policy-lag boundary {pending_step} predates the resumed '
                f'target update {target_optimizer_steps}; resume from the matching checkpoint'
            )
        if pending_step == target_optimizer_steps:
            pending_paths.append((pending_path, metadata))
        # A newer pending artifact can be left by a crash before the checkpoint
        # was saved. Replaying its boundary will replace it.
    if len(pending_paths) > 1:
        raise RuntimeError('multiple pending policy-lag boundaries match the current target checkpoint')
    if pending_paths:
        pending_path, pending_analysis = pending_paths[0]
        if state_digest(_target_lora_state_dict(model.target_model)) != pending_analysis['policy_checkpoint_id']:
            raise RuntimeError(f'resumed target does not match {pending_path}')
        if state_digest(model.draft_model.state_dict()) != pending_analysis['stale_draft_id']:
            raise RuntimeError(f'resumed stale draft does not match {pending_path}')
        for filename in ('draft_stale.pt', 'draft_fresh.pt'):
            if not (pending_path.parent / filename).is_file():
                raise RuntimeError(f'pending boundary needs {filename}: {pending_path}')
    # One rebuild on startup reconciles interrupted appends and removes results
    # newer than the restored training checkpoint. Never rebuild per boundary.
    analysis_per_response[:] = [row for row in analysis_per_response
                                if int(row['policy_step']) in analysis_completed]
    analysis_summaries[:] = [row for row in analysis_summaries if row.policy_step in analysis_completed]
    with analysis_io.measure():
        mark_durable_boundaries(analysis_root, target_optimizer_steps)
    analysis_completed, analysis_per_response, analysis_summaries = load_completed_results(analysis_root)
    analysis_per_response[:] = [row for row in analysis_per_response
                                if int(row['policy_step']) < target_optimizer_steps]
    analysis_summaries[:] = [row for row in analysis_summaries if row.policy_step < target_optimizer_steps]
    analysis_completed = {value for value in analysis_completed if value < target_optimizer_steps}
    with analysis_io.measure():
        write_results(analysis_root, analysis_per_response, analysis_summaries)
        cleanup_branch_checkpoints(analysis_root, target_optimizer_steps, keep=analysis_keep_branch_checkpoints)

training_data_generator = torch.Generator() if analysis_enabled else None
dataloader=DataLoader(
    QAs,
    collate_fn=TrainDataCollator(tokenizer=tokenizer, max_prompt_length=max_prompt_length),
    num_workers=num_workers,
    persistent_workers=persistent_workers and num_workers > 0,
    batch_size=batch_size,
    shuffle=True,
    drop_last=analysis_enabled,
    generator=training_data_generator,
)
if analysis_enabled and len(dataloader) == 0:
    raise ValueError(
        f'analysis needs at least {analysis_eval_prompts} training prompts; '
        f'only {len(QAs)} selected'
    )

def _training_epochs():
    epoch = start_epoch
    while not stop_requested and (epoch < num_epochs or (
            analysis_enabled and max_grpo_steps > 0 and target_optimizer_steps < max_grpo_steps)):
        before = target_optimizer_steps
        yield epoch
        if (analysis_enabled and max_grpo_steps > 0 and target_optimizer_steps == before
                and not (epoch == start_epoch and start_batch >= len(dataloader))):
            raise RuntimeError('entire epoch produced no usable target updates; check dataset/rewards/prompt length')
        epoch += 1


epoch_bar = tqdm(_training_epochs(), desc="Epoch (auto-extend to target step limit)", dynamic_ncols=True)
for epoch in epoch_bar:
    if training_data_generator is not None:
        # Reconstruct the same epoch permutation on resume before skipping its
        # consumed prefix. DataLoader must not consume the rollout's RNG.
        training_data_generator.manual_seed(trace_seed + epoch)
    
    batch_data['ignore_due_correct']=0
    batch_data['ignore_due_incorrect']=0
    batch_data['length_stdev'] = []
    batch_data['length_range'] = []
    batch_data['length_cv'] = []
    
    batch_bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Epoch {epoch + 1}/{num_epochs}",
        dynamic_ncols=True,
        leave=False,
    )
    for i,batch in enumerate(batch_bar):
        if epoch == start_epoch and i < start_batch:
            batch_bar.set_postfix(step=step, phase="resume_skip", refresh=False)
            continue
        
        if batch['input_ids'].shape[-1]>=max_length:
            batch=[]
            batch_bar.set_postfix(phase="skip_prompt_len", step=step, refresh=False)
            continue
        
        if None in batch['answers']:
            batch=[]
            batch_bar.set_postfix(phase="skip_none_answer", step=step, refresh=False)
            continue
        
        input_ids=batch['input_ids'].to('cuda')
        attention_mask=batch['attention_mask'].to('cuda')
        if analysis_enabled and input_ids.shape[0] != analysis_eval_prompts:
            raise RuntimeError('next-real-rollout analysis received an incomplete evaluation prompt batch')
        analysis_train_input_ids = input_ids.detach().cpu()
        analysis_train_attention_mask = attention_mask.detach().cpu()
        analysis_batch = {
            'input_ids': analysis_train_input_ids,
            'attention_mask': analysis_train_attention_mask,
        }
        analysis_base_draft = None
        analysis_base_optimizer = None
        analysis_old_teacher_logits = None
        analysis_old_policy_id = None
        analysis_old_policy_state = None
        if analysis_enabled:
            analysis_base_draft = {
                key: value.detach().cpu().clone()
                for key, value in model.draft_model.state_dict().items()
            }
            analysis_base_optimizer = _cpu_snapshot(optimizer_draft.state_dict())
            analysis_old_policy_state = {
                key: value.detach().cpu().clone()
                for key, value in _target_lora_state_dict(model.target_model).items()
            }
            analysis_old_policy_id = state_digest(analysis_old_policy_state)
            analysis_old_teacher_logits = _teacher_prefix_hidden(analysis_batch)
        pending_reflex_rows = None
        pending_reflex_timing = None
        pending_fresh_rows = None
        if pending_analysis is not None:
            pending_step = int(pending_analysis['policy_step'])
            pending_dir = Path(policy_lag_output_dir) / 'boundaries' / f'step_{pending_step}'
            if state_digest(_target_lora_state_dict(model.target_model)) != pending_analysis['policy_checkpoint_id']:
                raise RuntimeError(f'target changed before next rollout for boundary {pending_step}')
            if state_digest(model.draft_model.state_dict()) != pending_analysis['stale_draft_id']:
                raise RuntimeError(f'main draft is not the stale branch at boundary {pending_step}')
            if state_digest(optimizer_draft.state_dict()) != pending_analysis['stale_optimizer_id']:
                raise RuntimeError(f'main optimizer is not the stale branch at boundary {pending_step}')
            paired_rng = _analysis_rng_state()
            fresh_state = torch.load(pending_dir / 'draft_fresh.pt', map_location='cpu', weights_only=False)['draft_state_dict']
            pending_fresh_rows, _ = _evaluate_analysis_branch(
                'fresh', fresh_state, analysis_base_draft, analysis_batch, pending_analysis, epoch, i, paired_rng
            )
            pending_reflex_rows, pending_reflex_timing = _evaluate_analysis_branch(
                'reflex', analysis_base_draft, analysis_base_draft, analysis_batch, pending_analysis,
                epoch, i, paired_rng,
            )
            model.draft_model.load_state_dict(analysis_base_draft, strict=True)
            _restore_analysis_rng(paired_rng)
            if state_digest(model.draft_model.state_dict()) != pending_analysis['stale_draft_id']:
                raise RuntimeError(f'stale branch was not restored before real GRPO rollout {pending_step}')
            del fresh_state
        messages=batch['messages']
        answers=batch['answers']
        
        pending_modes = [(module, module.training) for module in model.modules()] if pending_analysis is not None else []
        if pending_analysis is not None:
            # Match the shadow rollout mode exactly. Restore train mode before
            # the real online draft backward pass below.
            model.eval()
        with torch.inference_mode():
            outputs=speculative_generate_in_prompt_batches(
            prompt_batch_size=analysis_eval_batch_size if analysis_enabled else 0,
            model=model,input_ids=input_ids,attention_mask=attention_mask,tokenizer=tokenizer,
            do_sample=True,max_length=max_length,repeated_generate_nums=repeated_generate_nums,temperature=temperature,top_p=top_p,
            verification_capacity=verification_capacity,
            max_draft_token_length=max_draft_token_length,
            max_draft_k=max_draft_k,
            max_verification_num=max_verification_num,
            min_draft_token_length=min_draft_token_length,
            draft_token_length_c=draft_token_length_c,
            return_all_draft_input=True,statistical_time=statistical_time,reflex_mode='off')
        if pending_analysis is not None:
            for module, mode in pending_modes:
                module.training = mode
        pending_stale_rows = (
            _analysis_rows_from_rollout(
                'stale', outputs, int(pending_analysis['policy_step']), epoch, i,
                used_for_grpo=True,
            ) if pending_analysis is not None else None
        )
        if pending_stale_rows is not None:
            rng_id = state_digest(paired_rng)
            for row in pending_stale_rows:
                row['initial_rng_id'] = rng_id
                row['evaluation_prompt_batch_id'] = state_digest(analysis_batch)
        
        prompt_length=input_ids.shape[-1]
        outputs['prompt_length']=prompt_length
        
        outputs['decoded_sequences']=[tokenizer.decode(x,skip_special_tokens=True) for x in outputs['generated_token_ids']]
        token_ids_length = [len(item) for item in outputs['generated_token_ids'] ]
        total_rollout_tokens = int(sum(token_ids_length))
        length_stdev = stdev(token_ids_length)
        length_range = max(token_ids_length) - min(token_ids_length)
        length_cv = length_stdev / mean(token_ids_length) 
        length_ave = mean(token_ids_length) 
        batch_data['generate_length_list'].extend(token_ids_length)
        batch_data['total_rollout_tokens']+=total_rollout_tokens
        
        draft_sparse_tv = None
        draft_sparse_kl = None
        draft_sparse_count = 0
        draft_update_committed = False
        if is_train_draft:
            torch.cuda.synchronize()
            draft_train_time_start=time.time()
            draft_loss1,draft_loss2,draft_sparse_tv,draft_sparse_kl,draft_sparse_count=training_draft_model(model,outputs,attention_mask)
            torch.cuda.synchronize()
            batch_data['draft_train_time_cost']+=time.time()-draft_train_time_start
            batch_data['last_draft_loss1'].append(draft_loss1)
            batch_data['last_draft_loss2'].append(draft_loss2)
            batch_data['draft_sparse_tv_sum'] += float(draft_sparse_tv) * int(draft_sparse_count)
            batch_data['draft_sparse_kl_sum'] += float(draft_sparse_kl) * int(draft_sparse_count)
            batch_data['draft_sparse_count'] += int(draft_sparse_count)
            draft_accumulated_step += 1
            if is_train_draft and draft_accumulated_step % draft_accumulation_steps == 0:
                optimizer_draft.step() 
                optimizer_draft.zero_grad(set_to_none=True)
                draft_step += 1
                draft_update_committed = True
    
        if draft_step % 1024 == 0 and step > 0 and is_train_draft:
            with open(f"{saved_statistics_dir}/{step}.pkl","wb") as f:
                pickle.dump(batch_data['generate_length_list'],f)
        
        generate_length=0
        for idx_batch in range(len(answers)):
            generate_length += outputs['max_sequence_length']
            rewards=[]
            new_messages=[]
            new_response_ids=[]
            for idx_k in range(repeated_generate_nums):
                idx_sequence=idx_batch*repeated_generate_nums+idx_k
                decoded_sequence=outputs['decoded_sequences'][idx_sequence]
                ground_truth=answers[idx_batch]
                
                new_message=deepcopy(messages[idx_batch])
                new_message.append({
                    "role": "assistant",
                    "content":decoded_sequence
                })
                
                format_reward=format_reward_func([decoded_sequence])
                answer_reward=accuracy_reward_func([decoded_sequence],[ground_truth])
                reward=0.2*format_reward[0]+answer_reward[0]
                
                rewards.append(reward)
                new_messages.append(new_message)
                new_response_ids.append((int(epoch), int(i), int(idx_batch), int(idx_k)))
            
            
            rewards=np.array(rewards) 
            if rewards.std()==0:
                
                if rewards[0]>=1.0:
                    batch_data['ignore_due_correct']+=1
                else:
                    batch_data['ignore_due_incorrect']+=1
                    
                continue
            
            std_rewards=(rewards-rewards.mean())/rewards.std()
            batch_data['messages']+=new_messages
            batch_data['rewards']+=rewards.tolist()
            batch_data['std_rewards']+=std_rewards.tolist()
            batch_data['response_ids']+=new_response_ids
            used_items+=1
            
        generate_length /= len(answers)
        
        batch_data['length_stdev'].append(length_stdev)
        batch_data['length_range'].append(length_range)
        batch_data['length_cv'].append(length_cv)
        batch_data['last_generate_time_cost'].append(outputs['total_time_cost'])
        batch_data['last_acc_length'].append(outputs['total_acc_length'])
        batch_data['last_decoded_token_num'].append(outputs['total_decoded_token_num'])
        accepted_draft_tokens = int(outputs.get('total_accepted_draft_tokens', 0))
        proposed_draft_tokens = int(outputs.get('total_proposed_draft_tokens', 0))
        batch_data['last_accepted_draft_tokens'].append(accepted_draft_tokens)
        batch_data['last_proposed_draft_tokens'].append(proposed_draft_tokens)
        batch_data['last_generate_length'].append(generate_length)
        batch_data['prefill_time_cost']+=outputs['prefill_time_cost']
        batch_data['target_time_cost']+=outputs['target_time_cost']
        batch_data['draft_time_cost']+=outputs['draft_time_cost']
        batch_data['check_time_cost']+=outputs['check_time_cost']
        
        batch_data['generate_time_cost']+=outputs['total_time_cost']
        batch_data['total_acc_length']+=outputs['total_acc_length']
        batch_data['total_decoded_token_num']+=outputs['total_decoded_token_num']
        batch_data['total_accepted_draft_tokens']+=accepted_draft_tokens
        batch_data['total_proposed_draft_tokens']+=proposed_draft_tokens
        batch_data['generate_length']+=generate_length
        trace_rollout_count += 1
        batch_data['trace_rollout_count'] = int(batch_data.get('trace_rollout_count', 0)) + 1
        source_grpo_step = used_items // max(1, batch_size * accumulation_steps)
        local_grpo_step = max(0, int(source_grpo_step - trace_start_step))
        rollout_log = {
            "phase": "rollout",
            "epoch": int(epoch + 1),
            "batch": int(i),
            "grpo_step": int(local_grpo_step),
            "source_grpo_step": int(source_grpo_step),
            "rollout_count": int(trace_rollout_count),
            "used_items": int(used_items),
            "draft_sparse_tv": None if draft_sparse_tv is None else float(draft_sparse_tv),
            "draft_sparse_kl": None if draft_sparse_kl is None else float(draft_sparse_kl),
            "draft_sparse_count": int(draft_sparse_count),
            "draft_update_committed": bool(draft_update_committed),
            "draft_updates_cumulative": int(draft_step - trace_start_draft_step),
            "drift_topk": int(drift_topk),
            "drift_temperature": float(drift_temperature),
            "draft_lr_multiplier": float(draft_lr_multiplier),
            "fastgrpo_ablation": bool(fastgrpo_ablation),
        }
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rollout_log) + '\n')
        batch=[]

        cur_acc_length = (
            batch_data['total_acc_length'] / max(batch_data['total_decoded_token_num'], 1)
        )
        cur_draft_acceptance_rate = (
            batch_data['total_accepted_draft_tokens'] /
            max(batch_data['total_proposed_draft_tokens'], 1)
        )
        batch_bar.set_postfix(
            step=step,
            acc=f"{cur_acc_length:.3f}",
            macc=f"{cur_draft_acceptance_rate:.3f}",
            gen=f"{outputs['total_time_cost'] / 60:.2f}m",
            pending=f"{len(batch_data['messages'])}/{batch_size * accumulation_steps}",
            phase="rollout",
            refresh=False,
        )

        if len(batch_data['messages']) == 0:
            if pending_analysis is not None:
                # This rollout cannot update the target, so it is not the
                # requested live measurement. Retry on the next usable batch.
                pending_dir = Path(policy_lag_output_dir) / 'boundaries' / f"step_{pending_analysis['policy_step']}"
                stale_checkpoint = torch.load(
                    pending_dir / 'draft_stale.pt', map_location='cpu', weights_only=False
                )
                model.draft_model.load_state_dict(stale_checkpoint['draft_state_dict'], strict=True)
                optimizer_draft.load_state_dict(stale_checkpoint['optimizer_state_dict'])
                if [float(group['lr']) for group in optimizer_draft.param_groups] != effective_draft_lrs:
                    raise RuntimeError('resumed stale branch restored an unexpected draft LR')
                optimizer_draft.zero_grad(set_to_none=True)
                if draft_update_committed:
                    draft_step -= 1
                    draft_accumulated_step -= 1
                with analysis_io.measure(), open(log_file, 'a', encoding='utf-8') as stream:
                    stream.write(json.dumps({
                        'phase': 'analysis_retry',
                        'policy_step': int(pending_analysis['policy_step']),
                        'epoch': int(epoch + 1),
                        'batch': int(i),
                        'reason': 'no_reward_variation_no_target_update',
                    }) + '\n')
            continue 
        
        text=tokenizer.apply_chat_template(batch_data['messages'],tokenize=False,add_generation_prompt=False)
        text=tokenizer(text,padding=False)
        loss_mask=[]
        
        for idx_message, message in enumerate(batch_data['messages']):
            prompt_text=tokenizer.apply_chat_template(message[:-1],tokenize=False,add_generation_prompt=True)
            prompt_text=tokenizer.encode(prompt_text)
            cur_loss_mask=[0]*(len(prompt_text)-1)+[1]*(len(text.input_ids[idx_message])-len(prompt_text)+1)
            loss_mask.append(cur_loss_mask)
            
        input_ids=text.input_ids
        attention_mask=text.attention_mask
        
        response_count = len(input_ids)
        if any(len(batch_data[field]) != response_count for field in
               ('messages', 'rewards', 'std_rewards', 'response_ids')):
            raise RuntimeError('GRPO per-response fields have different lengths before sorting')
        if len(set(tuple(item) for item in batch_data['response_ids'])) != response_count:
            raise RuntimeError('GRPO response identities are not unique')
        permutation, sorted_fields = reorder_response_fields(
            [len(row) for row in input_ids], {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask,
                **{field: batch_data[field] for field in
                   ('messages', 'rewards', 'std_rewards', 'response_ids')},
            }
        )
        input_ids = sorted_fields['input_ids']
        attention_mask = sorted_fields['attention_mask']
        loss_mask = sorted_fields['loss_mask']
        for field in ('messages', 'rewards', 'std_rewards', 'response_ids'):
            batch_data[field] = sorted_fields[field]
        if analysis_enabled:
            with analysis_io.measure(), open(log_file, 'a', encoding='utf-8') as stream:
                stream.write(json.dumps({
                    'phase': 'grpo_response_order',
                    'target_optimizer_steps_before': int(target_optimizer_steps),
                    'original_indices': permutation,
                    'response_ids': batch_data['response_ids'],
                    'advantages': batch_data['std_rewards'],
                }) + '\n')

        step = used_items // (batch_size * accumulation_steps)  
        batch_old_logps=[]
        batch_ref_logps=[]
        
        for grpo_iteration in range(grpo_iteration_num):
            if statistical_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            train_time_start=time.time()
            
            device=model.target_model.device
            groups = list(microbatch_index_groups(
                [len(row) for row in input_ids], max_training_token, max_training_padding_gap
            ))
            for microbatch_index, indices in enumerate(groups):
                padded_length = max(len(input_ids[j]) for j in indices)
                cur_input_ids = torch.tensor([
                    input_ids[j] + [0] * (padded_length - len(input_ids[j])) for j in indices
                ], device=device)
                cur_attention_mask = torch.tensor([
                    attention_mask[j] + [0] * (padded_length - len(attention_mask[j])) for j in indices
                ], device=device)
                cur_loss_mask = torch.tensor([
                    loss_mask[j] + [0] * (padded_length - len(loss_mask[j])) for j in indices
                ], device=device)
                cur_rewards = torch.tensor([
                    batch_data['std_rewards'][j] for j in indices
                ], device=device).unsqueeze(-1)
                if any(len(input_ids[j]) != len(attention_mask[j]) or
                       len(input_ids[j]) != len(loss_mask[j]) for j in indices):
                    raise RuntimeError('GRPO sequence and mask lengths differ')
                old_logps = None if grpo_iteration == 0 else batch_old_logps[microbatch_index]
                ref_logps = None if grpo_iteration == 0 else batch_ref_logps[microbatch_index]
                loss,abs_loss1,loss2,old_logps,ref_logps=compute_target_loss_and_backward(
                    model, cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards,
                    epsilon, beta, grpo_iteration, old_logps=old_logps,
                    ref_logps=ref_logps, chunk_size=logps_chunk_size,
                    loss_scale=1.0 / max(len(batch_data['messages']), 1),
                )
                if grpo_iteration == 0:
                    batch_old_logps.append(old_logps)
                    batch_ref_logps.append(ref_logps)
                del cur_input_ids, cur_attention_mask, cur_loss_mask, cur_rewards

                
            optimizer_target.step()
            optimizer_target.zero_grad(set_to_none=True)
            target_optimizer_steps += 1
            if pending_analysis is not None:
                if analysis_old_policy_id != pending_analysis['policy_checkpoint_id']:
                    raise RuntimeError('live rollout target did not match the pending policy-lag boundary')
                _finalize_next_rollout_analysis(
                    pending_analysis, pending_stale_rows, pending_fresh_rows,
                    pending_reflex_rows, pending_reflex_timing, epoch, i, target_optimizer_steps,
                    state_digest(analysis_batch),
                )
                pending_analysis = None

            if (
                analysis_enabled
                and _analysis_boundary(target_optimizer_steps)
                and target_optimizer_steps not in analysis_completed
                and (
                    max_grpo_steps == 0
                    or target_optimizer_steps < max_grpo_steps
                )
            ):
                analysis_step = target_optimizer_steps
                boundary_io_start = analysis_io.wall_ms
                analysis_main_rng = _analysis_rng_state()
                target_was_training = model.target_model.training
                draft_was_training = model.draft_model.training
                boundary_dir = Path(policy_lag_output_dir) / 'boundaries' / f'step_{analysis_step}'
                boundary_dir.mkdir(parents=True, exist_ok=True)
                current_policy_id = state_digest(_target_lora_state_dict(model.target_model))
                current_policy_state = {
                    key: value.detach().cpu().clone()
                    for key, value in _target_lora_state_dict(model.target_model).items()
                }
                target_digest_before_analysis = current_policy_id
                base_draft_digest = state_digest(analysis_base_draft)
                base_optimizer_digest = state_digest(analysis_base_optimizer)
                new_teacher_logits = _teacher_prefix_hidden(analysis_batch)
                teacher_tv = _teacher_tv_from_hidden(
                    analysis_old_teacher_logits,
                    new_teacher_logits,
                    analysis_train_attention_mask,
                )
                # Fresh R_{t+1}: same training prompts, current target, separate RNG;
                # use the *same base draft* as R_t to isolate the policy update.
                # It is never added to batch_data/the GRPO replay buffer.
                model.draft_model.load_state_dict(analysis_base_draft, strict=True)
                _seed_everything(trace_seed + analysis_step * 1009)
                with torch.inference_mode():
                    fresh_outputs = speculative_generate_in_prompt_batches(
                        prompt_batch_size=analysis_eval_batch_size,
                        model=model,
                        input_ids=analysis_train_input_ids.to('cuda'),
                        attention_mask=analysis_train_attention_mask.to('cuda'),
                        tokenizer=tokenizer,
                        do_sample=True,
                        max_length=max_length,
                        repeated_generate_nums=repeated_generate_nums,
                        temperature=temperature,
                        top_p=top_p,
                        verification_capacity=verification_capacity,
                        max_draft_token_length=max_draft_token_length,
                        max_draft_k=max_draft_k,
                        max_verification_num=max_verification_num,
                        min_draft_token_length=min_draft_token_length,
                        draft_token_length_c=draft_token_length_c,
                        return_all_draft_input=True,
                        statistical_time=False,
                        reflex_mode='off',
                    )
                old_available = count_eagle3_supervision(
                    outputs['all_draft_input_ids'], outputs['all_target_hidden_states'],
                    analysis_train_attention_mask, repeated_generate_nums,
                )
                fresh_available = count_eagle3_supervision(
                    fresh_outputs['all_draft_input_ids'], fresh_outputs['all_target_hidden_states'],
                    analysis_train_attention_mask, repeated_generate_nums,
                )
                configured_budget = int(args.analysis_training_token_budget)
                common_budget = min(old_available, fresh_available)
                if configured_budget > 0:
                    common_budget = min(common_budget, configured_budget)
                if common_budget <= 0:
                    raise RuntimeError(f'boundary {analysis_step} has no common valid supervised-token budget')
                with analysis_io.measure(), open(log_file, 'a', encoding='utf-8') as stream:
                    stream.write(json.dumps({
                        'phase': 'analysis_budget',
                        'policy_step': int(analysis_step),
                        'requested_training_token_budget': configured_budget,
                        'stale_available_valid_positions': old_available,
                        'fresh_available_valid_positions': fresh_available,
                        'actual_common_training_token_budget': common_budget,
                        'effective_draft_lrs': effective_draft_lrs,
                    }) + '\n')
                branch_update_steps = int(args.analysis_draft_update_steps)
                if common_budget < branch_update_steps:
                    raise RuntimeError(
                        f'training token budget {common_budget} is smaller than '
                        f'optimizer steps {branch_update_steps}'
                    )
                base_checkpoint = {
                    'format': 'fastgrpo_policy_lag_base_v5',
                    'policy_step': analysis_step,
                    'policy_t_id': analysis_old_policy_id,
                    'policy_t_plus_1_id': current_policy_id,
                    'policy_t_lora_state_dict': analysis_old_policy_state,
                    'policy_t_plus_1_lora_state_dict': current_policy_state,
                    'draft_state_dict': analysis_base_draft,
                    'optimizer_state_dict': analysis_base_optimizer,
                    'scheduler_state_dict': None,
                    'feature_layers': list(model.feature_layers),
                    'specforge': model.checkpoint_metadata(),
                }
                if analysis_keep_branch_checkpoints:
                    with analysis_io.measure():
                        _atomic_torch_save(base_checkpoint, boundary_dir / 'phi_base.pt')
                del base_checkpoint

                branch_states = {}
                stale_optimizer_state = None
                branch_losses = {}
                branch_gpu_times = {}
                for branch_name, branch_outputs, feature_policy in (
                    ('stale', outputs, analysis_old_policy_id),
                    ('fresh', fresh_outputs, current_policy_id),
                ):
                    model.draft_model.load_state_dict(analysis_base_draft, strict=True)
                    optimizer_draft.load_state_dict(deepcopy(analysis_base_optimizer))
                    if [float(group['lr']) for group in optimizer_draft.param_groups] != effective_draft_lrs:
                        raise RuntimeError(f'{branch_name} restored an unexpected draft LR')
                    if state_digest(model.draft_model.state_dict()) != base_draft_digest:
                        raise RuntimeError(f'{branch_name} did not start from phi_base')
                    if state_digest(optimizer_draft.state_dict()) != base_optimizer_digest:
                        raise RuntimeError(f'{branch_name} optimizer did not start from phi_base')
                    optimizer_draft.zero_grad(set_to_none=True)
                    consumed_tokens = 0
                    loss_totals = [0.0, 0.0]
                    update_events = []
                    for update_index in range(branch_update_steps):
                        step_budget = common_budget // branch_update_steps
                        if update_index < common_budget % branch_update_steps:
                            step_budget += 1
                        # CUDA events bracket forward/backward + AdamW, not
                        # checkpoint IO/target forward. Materialize after the
                        # branch, never synchronize inside each update.
                        update_start = torch.cuda.Event(enable_timing=True)
                        update_end = torch.cuda.Event(enable_timing=True)
                        update_start.record()
                        losses = training_draft_model(
                            model,
                            branch_outputs,
                            analysis_train_attention_mask.to('cuda'),
                            token_budget=step_budget,
                        )
                        consumed_tokens += int(losses[4])
                        loss_totals[0] += float(losses[0])
                        loss_totals[1] += float(losses[1])
                        optimizer_draft.step()
                        update_end.record()
                        update_events.append((update_start, update_end))
                        optimizer_draft.zero_grad(set_to_none=True)
                    update_events[-1][1].synchronize()
                    branch_gpu_times[branch_name] = sum(start.elapsed_time(end) for start, end in update_events)
                    if consumed_tokens != common_budget:
                        raise RuntimeError(
                            f'{branch_name} consumed {consumed_tokens} tokens, expected {common_budget}'
                        )
                    branch_states[branch_name] = {
                        key: value.detach().cpu().clone()
                        for key, value in model.draft_model.state_dict().items()
                    }
                    branch_optimizer_state = _cpu_snapshot(optimizer_draft.state_dict())
                    if branch_name == 'stale':
                        stale_optimizer_state = branch_optimizer_state
                    branch_losses[branch_name] = loss_totals
                    branch_checkpoint = {
                        'format': 'fastgrpo_policy_lag_branch_v5',
                        'branch': branch_name,
                        'policy_step': analysis_step,
                        'feature_policy_version': feature_policy,
                        'base_digest': base_draft_digest,
                        'draft_state_dict': branch_states[branch_name],
                        # Only stale's optimizer is needed for a pending retry.
                        'optimizer_state_dict': (branch_optimizer_state if branch_name == 'stale'
                                                 or analysis_keep_branch_checkpoints else None),
                        'scheduler_state_dict': None,
                        'actual_training_token_count': int(common_budget),
                        'requested_training_token_budget': configured_budget,
                        'effective_draft_lr': effective_draft_lrs[0],
                        'optimizer_steps': branch_update_steps,
                        'losses': branch_losses[branch_name],
                        'online_draft_update_gpu_ms': branch_gpu_times[branch_name],
                    }
                    with analysis_io.measure():
                        _atomic_torch_save(branch_checkpoint, boundary_dir / f'draft_{branch_name}.pt')
                    del branch_checkpoint
                    with analysis_io.measure(), open(log_file, 'a', encoding='utf-8') as stream:
                        stream.write(json.dumps({
                            'phase': 'analysis_branch_update',
                            'policy_step': int(analysis_step),
                            'branch': branch_name,
                            'requested_training_token_budget': configured_budget,
                            'actual_used_valid_positions': consumed_tokens,
                            'optimizer_steps': branch_update_steps,
                            'effective_draft_lr': effective_draft_lrs[0],
                            'base_draft_id': base_draft_digest,
                            'base_optimizer_id': base_optimizer_digest,
                            'online_draft_update_gpu_ms': branch_gpu_times[branch_name],
                        }) + '\n')

                # The stale branch is now the real training draft. Its next
                # rollout is the measured rollout *and* the next GRPO input.
                model.draft_model.load_state_dict(branch_states['stale'], strict=True)
                optimizer_draft.load_state_dict(stale_optimizer_state)
                if [float(group['lr']) for group in optimizer_draft.param_groups] != effective_draft_lrs:
                    raise RuntimeError('committed stale branch has an unexpected draft LR')
                draft_step += branch_update_steps - 1
                _restore_analysis_rng(analysis_main_rng)
                model.draft_model.train(draft_was_training)
                model.target_model.train(target_was_training)
                if state_digest(_target_lora_state_dict(model.target_model)) != target_digest_before_analysis:
                    raise RuntimeError('target policy changed during policy-lag evaluation')
                pending_analysis = {
                    'format': 'fastgrpo_policy_lag_pending_v5',
                    'policy_step': analysis_step,
                    'source_epoch': int(epoch + 1),
                    'source_batch': int(i),
                    'source_prompt_batch_id': state_digest(analysis_batch['input_ids']),
                    'old_policy_id': analysis_old_policy_id,
                    'policy_checkpoint_id': current_policy_id,
                    'base_draft_id': base_draft_digest,
                    'stale_draft_id': state_digest(branch_states['stale']),
                    'fresh_draft_id': state_digest(branch_states['fresh']),
                    'stale_optimizer_id': state_digest(stale_optimizer_state),
                    'common_training_token_budget': int(common_budget),
                    'requested_training_token_budget': configured_budget,
                    'stale_available_valid_positions': old_available,
                    'fresh_available_valid_positions': fresh_available,
                    'effective_draft_lr': effective_draft_lrs[0],
                    'optimizer_steps_per_branch': branch_update_steps,
                    'teacher_shift_tv': teacher_tv,
                    'evaluation_scope': 'next_real_grpo_rollout',
                    'stale_online_draft_update_gpu_ms': branch_gpu_times['stale'],
                    'fresh_online_draft_update_gpu_ms': branch_gpu_times['fresh'],
                    'analysis_io_wall_ms': analysis_io.wall_ms - boundary_io_start,
                }
                with analysis_io.measure():
                    atomic_json(boundary_dir / 'pending.json', pending_analysis)
                pending_analysis['analysis_io_wall_ms'] = analysis_io.wall_ms - boundary_io_start
                # The final timing scalar's own write is deliberately excluded.
                atomic_json(boundary_dir / 'pending.json', pending_analysis)
                del (
                    branch_states, branch_losses, stale_optimizer_state,
                    branch_optimizer_state, analysis_base_optimizer,
                    analysis_old_policy_state, current_policy_state,
                    analysis_old_teacher_logits, new_teacher_logits, fresh_outputs,
                )
            
            if statistical_time and torch.cuda.is_available():
                torch.cuda.synchronize()
            train_time_elapsed=time.time()-train_time_start
            batch_data['last_train_time_cost'].append(train_time_elapsed)
            batch_data['train_time_cost']+=train_time_elapsed
            batch_data['last_mean_rewards'].append(sum(batch_data['rewards'])/len(batch_data['rewards']))
            batch_data['mean_rewards']+=sum(batch_data['rewards'])/len(batch_data['rewards'])
            
            real_sample_num=sample_num*accumulation_steps
            last_accepted_draft_tokens=sum(batch_data['last_accepted_draft_tokens'][-real_sample_num:])
            last_proposed_draft_tokens=sum(batch_data['last_proposed_draft_tokens'][-real_sample_num:])
            draft_acceptance_rate=(
                batch_data['total_accepted_draft_tokens'] /
                max(batch_data['total_proposed_draft_tokens'], 1)
            )
            last_draft_acceptance_rate=last_accepted_draft_tokens / max(last_proposed_draft_tokens, 1)
            average_accept_length=(
                batch_data['total_acc_length'] /
                max(batch_data['total_decoded_token_num'], 1)
            )
            accepted_tokens_per_medusa_step=(
                batch_data['total_accepted_draft_tokens'] /
                max(batch_data['total_decoded_token_num'], 1)
            )
            
            avg_logs = {
                "phase":"target_train",
                "epoch":epoch+1,
                "step": step,
                "grpo_step": int(max(0, step - trace_start_step)),
                "source_grpo_step": int(step),
                "target_optimizer_steps": int(target_optimizer_steps),
                "rollout_count": int(trace_rollout_count),
                "used_items" : used_items ,
                "train_dataset_full_size": full_train_samples,
                "train_dataset_selected_size": selected_train_samples,
                "train_data_fraction": train_data_fraction,
                "train_subset_seed": train_subset_seed,
                "max_train_samples": max_train_samples,
                "logps_chunk_size": logps_chunk_size,
                f"length_range" : round(mean(batch_data['length_range']),4),
                f"length_cv" : round(mean(batch_data['length_cv']),4) ,
                f"length_stdev" : round(mean(batch_data['length_stdev']),4) ,  
                "grpo_iteration":grpo_iteration+1,
                "used_time": round((time.time()-start_time)/60, 3),
                f"last_{sample_num}_generate_time_cost":round(sum(batch_data['last_generate_time_cost'][-real_sample_num:])/60,3),
                f"last_{sample_num}_train_time_cost": round(sum(batch_data['last_train_time_cost'][-real_sample_num:]) / 60, 3),
                f"last_{sample_num}_acc_length":round(sum(batch_data['last_acc_length'][-real_sample_num:]) / sum(batch_data['last_decoded_token_num'][-real_sample_num:]),4),
                f"last_{sample_num}_draft_acceptance_rate":round(last_draft_acceptance_rate,4),
                f"last_{sample_num}_medusa_acceptance_rate":round(last_draft_acceptance_rate,4),
                f"last_{sample_num}_mean_rewards": round(sum(batch_data['last_mean_rewards'][-real_sample_num:]) / len(batch_data['last_mean_rewards'][-real_sample_num:]), 3),
                f"last_{sample_num}_mean_length": round(sum(batch_data['last_generate_length'][-real_sample_num:]) / len(batch_data['last_generate_length'][-real_sample_num:]), 3),
                
                "ignore_due_correct_cur_epoch":batch_data['ignore_due_correct'],
                "ignore_due_incorrect_cur_epoch":batch_data['ignore_due_incorrect'],                                
                "generate_time_cost":round(batch_data['generate_time_cost']/60,3),
                "average_acc_length":round(average_accept_length,4),
                "average_accept_length":round(average_accept_length,4),
                "accepted_tokens_per_medusa_step":round(accepted_tokens_per_medusa_step,4),
                "total_rollout_tokens":int(batch_data['total_rollout_tokens']),
                "total_accepted_draft_tokens":int(batch_data['total_accepted_draft_tokens']),
                "total_proposed_draft_tokens":int(batch_data['total_proposed_draft_tokens']),
                "total_accepted_medusa_tokens":int(batch_data['total_accepted_draft_tokens']),
                "total_proposed_medusa_tokens":int(batch_data['total_proposed_draft_tokens']),
                "draft_acceptance_rate":round(draft_acceptance_rate,4),
                "medusa_acceptance_rate":round(draft_acceptance_rate,4),
                "prefill_time_cost":round(batch_data['prefill_time_cost']/60,3),
                "target_time_cost":round(batch_data['target_time_cost']/60,3),
                "draft_time_cost":round(batch_data['draft_time_cost']/60,3),
                "train_time_cost":round(batch_data['train_time_cost']/60,3),
                "check_time_cost":round(batch_data['check_time_cost']/60,3),
                "mean_reward":round(batch_data['mean_rewards']/used_items,4),
                "draft_sparse_tv": None if draft_sparse_tv is None else float(draft_sparse_tv),
                "draft_sparse_kl": None if draft_sparse_kl is None else float(draft_sparse_kl),
                "draft_sparse_count": int(draft_sparse_count),
                "draft_update_committed": bool(draft_update_committed),
                "draft_updates_cumulative": int(draft_step - trace_start_draft_step),
                "draft_lr_multiplier": float(draft_lr_multiplier),
                "fastgrpo_ablation": bool(fastgrpo_ablation),
                
                "draft_train_time_cost":round(batch_data['draft_train_time_cost']/60,3) if is_train_draft else 0, 
                f"last_{sample_num}_draft_loss1":round(sum(batch_data['last_draft_loss1'][-real_sample_num:])/len(batch_data['last_draft_loss1'][-real_sample_num:]),4) if is_train_draft and draft_step > 0 else 0,
                f"last_{sample_num}_draft_loss2":round(sum(batch_data['last_draft_loss2'][-real_sample_num:])/len(batch_data['last_draft_loss2'][-real_sample_num:]),4) if is_train_draft and draft_step > 0 else 0 
            }

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(avg_logs) + '\n')

            postfix = {
                "step": step,
                "acc": avg_logs["average_accept_length"],
                "macc": avg_logs["medusa_acceptance_rate"],
                "gen": f"{avg_logs['last_' + str(sample_num) + '_generate_time_cost']:.2f}m",
                "train": f"{avg_logs['last_' + str(sample_num) + '_train_time_cost']:.2f}m",
                "reward": avg_logs["mean_reward"],
                "phase": "GRPO",
            }
            batch_bar.set_postfix(postfix, refresh=False)
            epoch_bar.set_postfix(postfix, refresh=False)
                
            torch.cuda.empty_cache()
            
        batch_data['messages'].clear()
        batch_data['rewards'].clear()
        batch_data['std_rewards'].clear()
        batch_data['response_ids'].clear()
        batch_old_logps.clear()
        batch_ref_logps.clear()

        if step%500==0 and step!=0:
            model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
            model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')

        checkpoint_step = target_optimizer_steps if analysis_enabled else step
        checkpoint_start_step = trace_start_target_optimizer_steps if analysis_enabled else trace_start_step
        if (
            save_checkpoint_steps > 0
            and (checkpoint_step - checkpoint_start_step) > 0
            and (checkpoint_step - checkpoint_start_step) % save_checkpoint_steps == 0
            and checkpoint_step != last_checkpoint_step
        ):
            save_training_checkpoint(
                checkpoint_dir,
                model=model,
                optimizer_target=optimizer_target,
                optimizer_draft=optimizer_draft,
                epoch=epoch,
                next_batch=i + 1,
                step=step,
                used_items=used_items,
                draft_step=draft_step,
                draft_accumulated_step=draft_accumulated_step,
                batch_data=batch_data,
                keep_last=keep_last_checkpoints,
                target_optimizer_steps=target_optimizer_steps if analysis_enabled else None,
            )
            last_checkpoint_step = checkpoint_step
            if analysis_enabled:
                with analysis_io.measure():
                    mark_durable_boundaries(analysis_root, target_optimizer_steps)
                    removed = cleanup_branch_checkpoints(
                        analysis_root, target_optimizer_steps, keep=analysis_keep_branch_checkpoints)
                if removed:
                    print(f'Cleaned {len(removed)} temporary branch checkpoints after durable GRPO save.')

        completed_grpo_steps = max(0, int(checkpoint_step - checkpoint_start_step))
        limit_progress = target_optimizer_steps if analysis_enabled else completed_grpo_steps
        if max_grpo_steps > 0 and limit_progress >= max_grpo_steps:
            stop_requested = True
            if checkpoint_step != last_checkpoint_step:
                save_training_checkpoint(
                    checkpoint_dir,
                    model=model,
                    optimizer_target=optimizer_target,
                    optimizer_draft=optimizer_draft,
                    epoch=epoch,
                    next_batch=i + 1,
                    step=step,
                    used_items=used_items,
                    draft_step=draft_step,
                    draft_accumulated_step=draft_accumulated_step,
                    batch_data=batch_data,
                    keep_last=keep_last_checkpoints,
                    target_optimizer_steps=target_optimizer_steps if analysis_enabled else None,
                )
                last_checkpoint_step = checkpoint_step
            if analysis_enabled:
                with analysis_io.measure():
                    mark_durable_boundaries(analysis_root, target_optimizer_steps)
                    cleanup_branch_checkpoints(analysis_root, target_optimizer_steps, keep=analysis_keep_branch_checkpoints)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    "phase": "trace_stop",
                    "grpo_step": int(completed_grpo_steps),
                    "source_grpo_step": int(step),
                    "target_optimizer_steps": int(target_optimizer_steps),
                    "rollout_count": int(trace_rollout_count),
                    "max_grpo_steps": int(max_grpo_steps),
                    "reason": "max_grpo_steps",
                }) + '\n')
            break

    if stop_requested:
        break
            

if pending_analysis is not None:
    raise RuntimeError(
        f"policy-lag boundary {pending_analysis['policy_step']} has no subsequent "
        'usable GRPO rollout; increase NUM_EPOCHS or TOTAL_POLICY_STEPS and resume'
    )

model.save_model(f"{saved_draft_model_dir}/step{step}.pth")
model.target_model.save_pretrained(f'{saved_model_dir}/step{step}')
if checkpoint_dir and not stop_requested:
    save_training_checkpoint(
        checkpoint_dir,
        model=model,
        optimizer_target=optimizer_target,
        optimizer_draft=optimizer_draft,
        epoch=num_epochs,
        next_batch=0,
        step=step,
        used_items=used_items,
        draft_step=draft_step,
        draft_accumulated_step=draft_accumulated_step,
        batch_data=batch_data,
        keep_last=keep_last_checkpoints,
        target_optimizer_steps=target_optimizer_steps if analysis_enabled else None,
    )
if analysis_enabled:
    with analysis_io.measure():
        mark_durable_boundaries(analysis_root, target_optimizer_steps)
        cleanup_branch_checkpoints(analysis_root, target_optimizer_steps, keep=analysis_keep_branch_checkpoints)
        analysis_completed, analysis_per_response, analysis_summaries = load_completed_results(analysis_root)
        # Final aggregation is IO; plotting is deliberately outside this timer.
        write_results(analysis_root, analysis_per_response, analysis_summaries)

final_average_accept_length = (
    batch_data['total_acc_length'] / max(batch_data['total_decoded_token_num'], 1)
)
final_medusa_acceptance_rate = (
    batch_data['total_accepted_draft_tokens'] /
    max(batch_data['total_proposed_draft_tokens'], 1)
)
final_accepted_tokens_per_medusa_step = (
    batch_data['total_accepted_draft_tokens'] /
    max(batch_data['total_decoded_token_num'], 1)
)
total_wall_time = time.time() - start_time
summary = {
    "run_name": version_name,
    "final_step": int(step),
    "target_optimizer_steps": int(target_optimizer_steps),
    "used_items": int(used_items),
    "draft_step": int(draft_step),
    "completed_grpo_steps": int(max(
        0,
        target_optimizer_steps - trace_start_target_optimizer_steps
        if analysis_enabled else step - trace_start_step,
    )),
    "max_grpo_steps": int(max_grpo_steps),
    "stopped_by_max_grpo_steps": bool(stop_requested),
    "rollout_count": int(trace_rollout_count),
    "draft_updates_committed": int(draft_step - trace_start_draft_step),
    "draft_sparse_tv": float(batch_data['draft_sparse_tv_sum']) / max(int(batch_data['draft_sparse_count']), 1),
    "draft_sparse_kl": float(batch_data['draft_sparse_kl_sum']) / max(int(batch_data['draft_sparse_count']), 1),
    "draft_sparse_count": int(batch_data['draft_sparse_count']),
    "draft_lr_multiplier": float(draft_lr_multiplier),
    "effective_draft_lrs": effective_draft_lrs,
    "fastgrpo_ablation": bool(fastgrpo_ablation),
    "train_dataset_full_size": int(full_train_samples),
    "train_dataset_selected_size": int(selected_train_samples),
    "dataset_path": str(args.dataset_path),
    "eval_dataset_path": str(args.eval_dataset_path),
    "train_data_fraction": float(train_data_fraction),
    "train_subset_seed": int(train_subset_seed),
    "max_train_samples": int(max_train_samples),
    "logps_chunk_size": int(logps_chunk_size),
    "total_generate_time_s": float(batch_data['generate_time_cost']),
    "total_train_time_s": float(batch_data['train_time_cost']),
    "total_draft_train_time_s": float(batch_data['draft_train_time_cost']) if is_train_draft else 0.0,
    "total_wall_time_s": float(total_wall_time),
    "analysis_io_wall_ms": analysis_io.wall_ms if analysis_enabled else 0.0,
    "corrected_wall_time_s": float(total_wall_time - analysis_io.wall_ms / 1000.0) if analysis_enabled else float(total_wall_time),
    "total_rollout_tokens": int(batch_data['total_rollout_tokens']),
    "generation_tokens_per_s": (
        float(batch_data['total_rollout_tokens']) /
        max(float(batch_data['generate_time_cost']), 1e-9)
    ),
    "average_accept_length": float(final_average_accept_length),
    "average_acc_length": float(final_average_accept_length),
    "accepted_tokens_per_medusa_step": float(final_accepted_tokens_per_medusa_step),
    "medusa_acceptance_rate": float(final_medusa_acceptance_rate),
    "draft_acceptance_rate": float(final_medusa_acceptance_rate),
    "total_acc_length": int(batch_data['total_acc_length']),
    "total_decoded_token_num": int(batch_data['total_decoded_token_num']),
    "total_verify_rounds": int(batch_data['total_decoded_token_num']),
    "total_accepted_draft_tokens": int(batch_data['total_accepted_draft_tokens']),
    "total_proposed_draft_tokens": int(batch_data['total_proposed_draft_tokens']),
    "total_accepted_medusa_tokens": int(batch_data['total_accepted_draft_tokens']),
    "total_proposed_medusa_tokens": int(batch_data['total_proposed_draft_tokens']),
    "mean_reward": (
        float(batch_data['mean_rewards']) / max(int(used_items), 1)
    ),
    "ignore_due_correct": int(batch_data['ignore_due_correct']),
    "ignore_due_incorrect": int(batch_data['ignore_due_incorrect']),
    "metrics_jsonl": str(log_file),
    "summary_json": str(summary_file),
    "saved_model_dir": f"{saved_model_dir}/step{step}",
    "saved_draft_model_dir": f"{saved_draft_model_dir}/step{step}.pth",
    "saved_statistics_dir": str(saved_statistics_dir),
    "checkpoint_dir": str(checkpoint_dir),
}
summary_text = json.dumps(summary, indent=2, ensure_ascii=True)
with open(summary_file, "w", encoding="utf-8") as f:
    f.write(summary_text + "\n")
summary_txt = os.path.join(os.path.dirname(summary_file), "summary.txt")
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write(summary_text + "\n")
print(summary_text)
if analysis_enabled:
    # No rendering in the training/boundary loop.
    from policy_lag_analysis import plot_results
    from dataclasses import asdict
    plot_status = plot_results([asdict(item) for item in analysis_summaries], analysis_root / 'aal_policy_lag.png')
    atomic_json(analysis_root / 'plot_status.json', plot_status)

"""Reference Qwen2.5-3B defaults with explicit one-optimizer-step overrides."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import json

@dataclass
class Config:
    model: str = ''
    draft_checkpoint: str = ''
    draft_target_config: str = ''
    train_path: str = ''
    test_path: str = ''
    target_adapter: str = ''
    output_dir: str = ''
    train_steps: int = 200
    eval_steps: tuple = (20,50,100,150,200)
    confirmation_steps: tuple = (100,200)
    seed: int = 42
    eval_subset_seed: int = 2026
    batch_size: int = 8
    responses_per_prompt: int = 8
    accumulation_steps: int = 1
    draft_accumulation_steps: int = 1
    grpo_iteration_num: int = 1
    target_lr: float = 1e-6
    draft_lr: float = 1e-4
    opd_projector_lr: float | None = None
    beta: float = .04
    epsilon: float = .1
    temperature: float = 1.
    top_p: float = .95
    top_k: int | None = None
    max_length: int = 2048
    max_prompt_length: int = 2048
    max_training_token: int = 1024
    max_training_padding_gap: int = 4096
    verification_capacity: int = 512
    max_verification_num: int = 160
    max_draft_token_length: int = 5
    max_draft_k: int = 8
    min_draft_token_length: int = 3
    draft_token_length_c: float = .75
    opd_rank: int = 8
    opd_topk: int = 16
    opd_fast_lr: float = .01
    opd_visited_weight: float = 1.
    opd_frontier_weight: float = 1.
    opd_update_stream: bool = True
    opd_sampler_mode: str = 'finite'
    opd_proposal_mode: str = 'auto'
    opd_dense_implementation: str = 'auto'
    opd_proposal_profile: str = ''
    eval_max_new_tokens: int = 256
    bootstrap_samples: int = 2000
    dtype: str = 'bf16'
    attention_implementation: str = 'sdpa'
    replay_hidden_atol: float = .125
    replay_hidden_rtol: float = .05
    replay_distribution_tv: float = .02
    zero_update_steps: tuple = (20,)
    smoke: bool = False
    smoke_eval_prompts: int = 2
    save_every: int = 1

    def validate(self):
        if self.accumulation_steps != 1 or self.draft_accumulation_steps != 1 or self.grpo_iteration_num != 1:
            raise ValueError('This causal protocol requires accumulation_steps=draft_accumulation_steps=grpo_iteration_num=1')
        if (self.batch_size,self.responses_per_prompt) != (8,8) and not self.smoke:
            raise ValueError('Production requires 8 distinct train prompts x 8 responses per learner')
        if self.smoke and self.smoke_eval_prompts not in (2,3,4): raise ValueError('Smoke uses 2-4 test prompts')
        if self.train_steps < 1 or self.save_every < 1: raise ValueError('train_steps/save_every must be positive')
        if len(set(self.eval_steps)) != len(self.eval_steps): raise ValueError('Duplicate eval steps')
        for name in ('eval_steps','confirmation_steps','zero_update_steps'):
            values = getattr(self,name)
            if any(s < 1 or s > self.train_steps for s in values):
                raise ValueError(f'{name} must be within completed target steps 1..{self.train_steps}; override schedule explicitly')
        if not set(self.zero_update_steps) <= set(self.eval_steps): raise ValueError('Zero-update controls must be measured boundaries')
        if not set(self.confirmation_steps) <= set(self.eval_steps): raise ValueError('Confirmations must be scheduled eval boundaries')
        if self.eval_subset_seed != 2026: raise ValueError('Preregistered eval_subset_seed is 2026')
        if self.eval_max_new_tokens < 2: raise ValueError('eval_max_new_tokens must permit at least one verification round (>=2)')
        if self.max_length < 4 or self.max_prompt_length < 1: raise ValueError('Invalid sequence/prompt lengths')
        if self.temperature <= 0 or not 0 <= self.top_p <= 1: raise ValueError('Invalid decoding distribution')
        if self.top_k is not None and self.top_k < 1: raise ValueError('top_k must be positive or null')
        if self.opd_topk < self.max_draft_k: raise ValueError('OPD topk must cover draft K')
        if not 1 <= self.opd_rank <= 64 or self.opd_fast_lr < 0: raise ValueError('Invalid OPD rank/lr')
        if self.opd_sampler_mode not in ('strict','finite'): raise ValueError('Invalid OPD sampler mode')
        if self.dtype not in ('bf16','fp16'): raise ValueError('Production generation requires bf16/fp16')
        if self.bootstrap_samples < 1: raise ValueError('bootstrap_samples must be positive')
        return self

    def n_eval(self, step):
        return self.smoke_eval_prompts if self.smoke else (64 if step in self.confirmation_steps else 16)

    def to_dict(self):
        # Manifest identity must survive JSON serialization before a later CLI
        # invocation compares it with the current configuration.
        payload = asdict(self)
        for name in ('eval_steps','confirmation_steps','zero_update_steps'):
            payload[name] = list(payload[name])
        return payload


def load_config(path):
    payload = json.loads(Path(path).read_text()) if path else {}
    for k in ('eval_steps','confirmation_steps','zero_update_steps'):
        if k in payload: payload[k] = tuple(payload[k])
    return Config(**payload)

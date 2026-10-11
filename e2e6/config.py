from dataclasses import dataclass,asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

@dataclass
class Config:
    model:str='/workspace/storage-shared/models/Qwen2.5-3B-Instruct'
    train_path:str='/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/train.parquet'
    test_path:str='/workspace/storage-shared/nlp/minhpn19/data/simplelr_abel_level3to5/test.parquet'
    draft_checkpoint:str='/workspace/storage-shared/nlp/minhpn19/SpecNaacl/outputs/pretrain/qwen25_3b/latest_checkpoint'
    draft_target_config:str=''
    output_dir:str=str(ROOT/'outputs/e2_e6_fastgrpo')
    seed:int=42
    eval_seed:int=2026
    train_steps:int=500
    batch_size:int=8
    responses_per_prompt:int=8
    accumulation_steps:int=4  # upstream reporting setting; see README_E2_E6
    draft_accumulation_steps:int=1
    target_lr:float=1e-6
    draft_lr:float=1e-4
    beta:float=.04
    epsilon:float=.1
    temperature:float=1.
    top_p:float=.95
    top_k:int|None=None
    max_length:int=2048
    max_prompt_length:int=2048
    max_training_token:int=1024
    max_training_padding_gap:int=4096
    draft_ce_chunk_size:int=128
    verification_capacity:int=512
    max_verification_num:int=160
    max_draft_token_length:int=5
    max_draft_k:int=8
    min_draft_token_length:int=3
    draft_token_length_c:float=.75
    dtype:str='bf16'
    attention_implementation:str='sdpa'
    target_gradient_checkpointing:bool=True
    eval_rollouts:int=20
    bootstrap_samples:int=2000
    save_every:int=10
    max_rollout_attempts:int=100000
    require_b200:bool=False
    smoke:bool=False

    def to_dict(self):return asdict(self)
    def validate(self):
        if not 1<=self.train_steps<=500:raise ValueError('Target steps must be in 1..500')
        if not self.smoke and (self.batch_size,self.responses_per_prompt)!=(8,8):raise ValueError('Production requires 8 prompts x 8 responses')
        if self.draft_accumulation_steps!=1:raise ValueError('Reference online draft accumulation is 1')
        if min(self.batch_size,self.responses_per_prompt,self.eval_rollouts,self.bootstrap_samples,self.save_every)<1:
            raise ValueError('Positive batch/rollout/bootstrap/checkpoint settings required')
        if self.save_every>10:raise ValueError('Resume checkpoint interval must be <=10 completed optimizer steps')
        if self.temperature<=0 or not 0<self.top_p<=1:raise ValueError('Invalid sampling settings')
        if self.dtype not in ('bf16','fp16'):raise ValueError('Use bf16/fp16 CUDA generation')
        # Validation-only pretrained paths are never silently accepted.
        if any(part in ('validation','.specnaacl-validation','smoke-latest') for part in Path(self.draft_checkpoint).parts):
            raise ValueError('A production pretrained draft is required; validation-only draft paths are rejected')


def generation_kwargs(c,training=False):
    return dict(do_sample=True,repeated_generate_nums=c.responses_per_prompt,temperature=c.temperature,
        top_p=c.top_p,top_k=c.top_k,verification_capacity=c.verification_capacity,
        max_draft_token_length=c.max_draft_token_length,max_draft_k=c.max_draft_k,
        max_verification_num=c.max_verification_num,min_draft_token_length=c.min_draft_token_length,
        draft_token_length_c=c.draft_token_length_c,statistical_time=False,return_all_draft_input=training,
        max_length=c.max_length)

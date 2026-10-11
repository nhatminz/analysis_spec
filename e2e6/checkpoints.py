"""Atomic resumable generations, immutable milestone hardlinks and hash validation."""
from pathlib import Path
import json
import os
import time
import torch
from motivation.state import atomic_json,atomic_torch,digest,file_hash

FORMAT='pure_fastgrpo_e2_e6_v1'
MILESTONES=(30,100,200,500)


def _link(source,target):
    temporary=target.with_name(target.name+'.tmp')
    if temporary.exists():temporary.unlink()
    os.link(source,temporary);os.replace(temporary,target)


def verify(path,load=True):
    path=Path(path)
    info=json.loads(path.with_suffix('.json').read_text())
    if path.stat().st_size!=info['size'] or file_hash(path)!=info['sha256']:
        raise ValueError(f'Checkpoint integrity failed: {path}')
    if not load:return info
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!=FORMAT or state['policy_step']!=info['policy_step'] or state['attempt']!=info['attempt']:
        raise ValueError('Checkpoint format/counters differ from integrity sidecar')
    if digest(state['target_lora'])!=info['target_lora_sha256'] or digest(state['draft_weights'])!=info['draft_weights_sha256']:
        raise ValueError('Checkpoint weight digest mismatch')
    return state


def save(directory,state,milestone=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    state=dict(state,format=FORMAT)
    generation=directory/f'resume_step_{state["policy_step"]:06d}_attempt_{state["attempt"]:08d}_{time.time_ns()}.pt'
    atomic_torch(generation,state)
    info=dict(format=FORMAT,policy_step=state['policy_step'],attempt=state['attempt'],size=generation.stat().st_size,
              sha256=file_hash(generation),target_lora_sha256=digest(state['target_lora']),draft_weights_sha256=digest(state['draft_weights']))
    atomic_json(generation.with_suffix('.json'),info)
    verify(generation,load=False)
    # Commit BEFORE publishing a milestone. A crash must never publish a
    # milestone whose completed optimizer step the resume pointer would replay.
    atomic_json(directory/'latest.json',dict(path=generation.name,**info))
    if milestone:
        target=directory/f'step_{state["policy_step"]:06d}.pt'
        if target.exists():
            # A later skipped target update may have updated the draft at this
            # same policy step. Preserve the FIRST finalized milestone exactly.
            verify(target,load=False)
        else:
            atomic_json(target.with_suffix('.json'),info);_link(generation,target)
    # latest.json is the sole durable commit pointer. Aliases can be repaired.
    _link(generation,directory/'latest.pt');atomic_json(directory/'latest.integrity.json',info)
    # Preserve newest two resumable generations; NEVER remove step_*.pt milestones.
    generations=sorted(directory.glob('resume_step_*_attempt_*.pt'),key=lambda p:p.stat().st_mtime_ns)
    for old in generations[:-2]:
        old.unlink();old.with_suffix('.json').unlink(missing_ok=True)
    return generation


def latest(directory):
    directory=Path(directory);info=json.loads((directory/'latest.json').read_text())
    path=directory/info['path'];actual=verify(path,load=False)
    if actual['sha256']!=info['sha256']:raise ValueError('Latest commit pointer differs from checkpoint')
    return path


def repair_milestone(directory,path):
    """Recover a crash after latest commit but before the milestone hardlink."""
    info=verify(path,load=False)
    if info['policy_step'] in MILESTONES:
        target=Path(directory)/f'step_{info["policy_step"]:06d}.pt'
        if not target.exists():
            atomic_json(target.with_suffix('.json'),info);_link(Path(path),target)


def select(directory,steps):
    paths=[];missing=[]
    for step in sorted(set(steps)):
        path=Path(directory)/f'step_{step:06d}.pt'
        if not path.exists():missing.append(step);continue
        info=verify(path,load=False)
        if info['policy_step']!=step:raise ValueError('Milestone filename/step mismatch')
        paths.append(path)
    return paths,missing

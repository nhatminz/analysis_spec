"""Read-only checkpoint decoding; A must belong to this exact checkpoint."""
from pathlib import Path
import torch
from motivation.state import digest, file_hash
from e2e6.checkpoints import FORMAT, verify


def projector_training_evidence(state, projector, optimizer):
    evidence=[]
    initial=state.get('opd_projector_initial')
    if initial is not None and digest(initial)!=digest(projector):evidence.append('A differs from this checkpoint trajectory initial A')
    for moment in (optimizer or {}).get('state',{}).values():
        avg=moment.get('exp_avg')
        if torch.is_tensor(avg) and tuple(avg.shape)==tuple(projector.shape) and float(moment.get('step',0))>0:
            evidence.append('saved Adam moments and positive update counter for A-shaped parameter')
            break
    return evidence


def decode(state, step, mode):
    if mode not in ('learned','initialized'):raise ValueError('Projector mode must be learned or explicitly initialized')
    kind=state.get('format');projector=None;optimizer=None
    if kind==FORMAT:
        actual=state['policy_step'];draft=dict(state['draft_weights']);target=state['target_lora']
    elif kind=='simplelr_opd_policy_lag_a1a2_v2':
        if state.get('training_trajectory')!='OPD-driven':raise ValueError('SPARK checkpoint trajectory is not declared')
        actual=state['policy_step'];draft=dict(state['reflex']['weights']);target=state['target_lora']
        optimizer=state['reflex'].get('optimizer')
    elif kind=='opd_fastgrpo_checkpoint_v5':
        if state.get('method')!='opd_reflex':raise ValueError('Expected a SPARK-trained checkpoint')
        actual=state['step'];draft=dict(state['draft_model']);target=state['target_lora']
        optimizer=state.get('optimizer_draft');projector=state.get('opd_projector')
    else:raise ValueError(f'Unsupported paired evaluation checkpoint format: {kind!r}')
    if int(actual)!=int(step):raise ValueError('Requested checkpoint label differs from saved counter')
    in_draft=draft.pop('opd_projector',None)
    if projector is None:projector=in_draft
    elif in_draft is not None and digest(projector)!=digest(in_draft):raise ValueError('Conflicting copies of checkpoint A')
    if not draft or not target:raise ValueError('Checkpoint is missing frozen target/draft weights')
    if any('opd_' in key for key in draft):raise ValueError('Unexpected correction state in backbone checkpoint')
    for value in list(draft.values())+list(target.values()):
        if not torch.is_tensor(value) or not torch.isfinite(value).all():raise ValueError('Nonfinite/non-tensor model checkpoint weight')
    if mode=='learned':
        if projector is None:
            raise ValueError('No learned A in this checkpoint. Pure FastGRPO requires explicit --projector-mode initialized (exploratory).')
        if projector.ndim!=2 or not torch.isfinite(projector).all():raise ValueError('Invalid checkpoint projector A')
        evidence=projector_training_evidence(state,projector,optimizer)
        if int(actual)<=0 or not evidence:raise ValueError('Checkpoint A has no verifiable training provenance; cannot label it learned')
        provenance=dict(mode='learned',interpretation='Within-checkpoint learned SPARK projector',
                        training_evidence=evidence,A_sha256=digest(projector),rank=projector.shape[1])
    else:
        if projector is not None:raise ValueError('Checkpoint contains A; use its learned projector rather than discarding or replacing it')
        provenance=dict(mode='initialized',interpretation='EXPLORATORY: deterministic untrained A, not fully trained SPARK',
                        initialization='head-aligned CPU QR from the frozen target LM head, no random draws',A_sha256=None)
    return dict(draft=draft,target=target,projector=projector,provenance=provenance,
                checkpoint_format=kind,checkpoint_step=int(actual),
                target_lora_sha256=digest(target),draft_backbone_sha256=digest(draft))


def load(path,step,mode):
    path=Path(path).resolve(strict=True)
    before=file_hash(path)
    if path.with_suffix('.json').is_file():
        import json
        sidecar=json.loads(path.with_suffix('.json').read_text())
    else:sidecar={}
    if sidecar.get('format')==FORMAT:state=verify(path)
    else:state=torch.load(path,map_location='cpu',weights_only=False)
    result=decode(state,step,mode)
    if file_hash(path)!=before:raise ValueError('Checkpoint changed while being read')
    result['identity']=dict(path=str(path),sha256=before,size=path.stat().st_size,
        checkpoint_step=step,format=result['checkpoint_format'],target_lora_sha256=result['target_lora_sha256'],
        draft_backbone_sha256=result['draft_backbone_sha256'],projector=result['provenance'])
    return result

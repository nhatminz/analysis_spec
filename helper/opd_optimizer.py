"""Optional projector LR, preserving optimizer moments on old single-group resumes."""
import copy
import torch


def draft_optimizer(draft,lr,projector_lr=None):
    if projector_lr is None:return torch.optim.AdamW(draft.parameters(),lr=lr)
    if projector_lr<0:raise ValueError('OPD_PROJECTOR_LR must be nonnegative')
    projector=draft.opd_projector
    return torch.optim.AdamW([
        {'params':[p for p in draft.parameters() if p is not projector],'lr':lr,'name':'draft'},
        {'params':[projector],'lr':projector_lr,'name':'opd_projector'}],lr=lr)


def load_draft_optimizer(optimizer,state,draft):
    if len(state['param_groups'])==2 and len(optimizer.param_groups)==1:
        # Unset OPD_PROJECTOR_LR on resume means preserve the saved separate LR,
        # not discard moments or silently collapse A into the draft LR group.
        if [g.get('name') for g in state['param_groups']]!=['draft','opd_projector']:
            raise ValueError('unrecognized saved draft optimizer groups')
        projector=draft.opd_projector
        optimizer.param_groups[0]['params']=[p for p in draft.parameters() if p is not projector]
        optimizer.add_param_group({'params':[projector],'name':'opd_projector'})
    if len(state['param_groups'])==len(optimizer.param_groups):
        optimizer.load_state_dict(state);return
    if len(state['param_groups'])!=1 or len(optimizer.param_groups)!=2:
        raise ValueError('unsupported draft optimizer group migration')
    old=state['param_groups'][0]
    parameters=list(draft.parameters())
    if len(old['params'])!=len(parameters):raise ValueError('checkpoint draft parameters differ')
    ids={id(p):idx for p,idx in zip(parameters,old['params'])}
    migrated=copy.deepcopy(state);migrated['param_groups']=[]
    for group in optimizer.param_groups:
        new=dict(old,params=[ids[id(p)] for p in group['params']],name=group['name'])
        if group['name']=='opd_projector':new['lr']=group['lr']
        migrated['param_groups'].append(new)
    optimizer.load_state_dict(migrated)

"""Matched-sequence FastGRPO teacher forcing, including the soft CE teacher.

For an unpadded original prompt P and production draft history D of length L:
    C = [P[0]] + D
    D[i] = C[i+1], captured_feature[i] = T(C[:i+1]).last_hidden[i]
The teacher input is C[:-1], NOT D. The upstream loss compares predicted
feature[i] to feature[i+1], and predicted logits[i] to
softmax(frozen_full_head(feature[i+1])). Its mask is i >= len(P), i < L-1.

Distributions are stored losslessly in factored form (hidden features + a
checksum-verified frozen head), avoiding O(trajectories * tokens * vocabulary)
checkpoint space. Every old replay position is checked, including prompt
features; both hidden states and the full-softmax CE distribution are gated.
"""
from dataclasses import dataclass
from copy import deepcopy
import torch
from motivation.state import cpu_copy, digest

@dataclass
class CapturedSequence:
    example_index: int
    prompt_id: str
    prompt_length: int
    original_prompt_ids: torch.Tensor
    context_ids: torch.Tensor
    draft_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    loss_mask: torch.Tensor
    teacher_features: torch.Tensor

    def validate(self):
        l=len(self.draft_input_ids)
        if len(self.context_ids)!=l+1 or self.teacher_features.shape[0]!=l:
            raise ValueError('Teacher/draft shifted sequence dimensions differ')
        if not torch.equal(self.context_ids[1:], self.draft_input_ids): raise ValueError('Lost one-token draft shift')
        if not torch.equal(self.context_ids[:self.prompt_length],self.original_prompt_ids):
            raise ValueError('Original prompt context was altered/lost')
        if not torch.equal(self.position_ids,torch.arange(l)):raise ValueError('Accepted context position alignment differs')
        if not torch.equal(self.attention_mask,torch.ones(l,dtype=torch.long)):raise ValueError('Captured unpadded context has gaps')
        expected=(torch.arange(l)>=self.prompt_length).long()
        if not torch.equal(self.loss_mask,expected):raise ValueError('Production loss mask differs')
        # Early EOS can legitimately leave no reference loss positions. Keep
        # the exact context in the trace; the objective excludes this example
        # from BOTH its numerator and normalization, identically for Fresh.
        return self

    def invariant(self):
        return digest({k:v for k,v in vars(self).items() if k!='teacher_features'})


def capture(outputs, input_ids, attention_mask, prompt_ids, repeats):
    sequences=[]
    features=outputs['all_draft_input_states'];ids=outputs['all_draft_input_ids']
    if len(features)!=len(prompt_ids)*repeats or len(ids)!=len(features):raise ValueError('Incorrect rollout response count')
    for i,(d,h) in enumerate(zip(ids,features)):
        j=i//repeats
        prompt=input_ids[j][attention_mask[j].bool()].detach().cpu().clone()
        d=cpu_copy(d);h=cpu_copy(h);l=len(d)
        s=CapturedSequence(i,prompt_ids[j],len(prompt),prompt,torch.cat((prompt[:1],d)),d,
                           torch.ones(l,dtype=torch.long),torch.arange(l),
                           (torch.arange(l)>=len(prompt)).long(),h).validate()
        # Generated IDs remain a distinct field: initial-root EOS is filtered
        # earlier than history by upstream. Never rebuild supervision by decoding.
        response=torch.tensor(outputs['generated_token_ids'][i],dtype=torch.long)
        if not torch.equal(s.context_ids[len(prompt):len(prompt)+len(response)],response):
            raise ValueError('Accepted response IDs do not align with captured original context')
        sequences.append(s)
    return sequences


def base_model(target):
    return target.get_base_model() if hasattr(target,'get_base_model') else target


@torch.no_grad()
def teacher_features(target, sequence):
    base=base_model(target);device=base.device
    ids=sequence.context_ids[:-1].unsqueeze(0).to(device)
    positions=sequence.position_ids.unsqueeze(0).to(device)
    l=ids.shape[-1]
    # Explicit causal attention matches production target's direct decoder path.
    mask=torch.full((l,l),torch.finfo(base.dtype).min,dtype=base.dtype,device=device).triu(1)[None,None]
    with torch.autocast(device_type=device.type,dtype=base.dtype,enabled=device.type=='cuda'):
        result=base.model(input_ids=ids,attention_mask=mask,position_ids=positions,use_cache=False,return_dict=True)
    h=result.last_hidden_state[0].detach().cpu().clone()
    if h.shape!=sequence.teacher_features.shape:raise ValueError('Teacher-forced target hidden dimension mismatch')
    return h


@torch.no_grad()
def distribution(head, features):
    device=head.weight.device
    return head(features.to(device=device,dtype=head.weight.dtype)).float().softmax(-1)


@torch.no_grad()
def replay_old_gate(target, sequences, config, teacher_trace=None):
    head=base_model(target).lm_head
    diagnostics=[]
    replayed_batch=teacher_trace.replay(target) if teacher_trace is not None else None
    if replayed_batch is not None and len(replayed_batch)!=len(sequences):raise RuntimeError('Teacher trace response count mismatch')
    for index,sequence in enumerate(sequences):
        sequence.validate();actual=sequence.teacher_features;replayed=replayed_batch[index] if replayed_batch is not None else teacher_features(target,sequence)
        if actual.shape!=replayed.shape:raise RuntimeError('Teacher trace feature alignment/length mismatch')
        if not torch.isfinite(replayed).all():raise RuntimeError('Nonfinite replay teacher')
        error=(actual.float()-replayed.float()).abs()
        bound=config.replay_hidden_atol + config.replay_hidden_rtol * actual.float().abs()
        if (error>bound).any():
            raise RuntimeError(f'Old-target hidden replay mismatch for {sequence.prompt_id}: '
                               f'max abs={error.max().item():.6g}. Fresh is invalid; no estimate will be published.')
        maximum_tv=0.;tv_sum=0.;positions=0
        for start in range(0,len(actual),16):
            p=distribution(head,actual[start:start+16]);q=distribution(head,replayed[start:start+16])
            tv=.5*(p-q).abs().sum(-1);maximum_tv=max(maximum_tv,float(tv.max()));tv_sum+=float(tv.sum());positions+=len(tv)
        if maximum_tv > config.replay_distribution_tv:
            raise RuntimeError(f'Old-target full-softmax distribution replay mismatch for {sequence.prompt_id}: '
                               f'max TV={maximum_tv:.6g} > {config.replay_distribution_tv}; Fresh is invalid')
        diagnostics.append(dict(example_index=sequence.example_index,prompt_id=sequence.prompt_id,
                                hidden_max_abs=float(error.max()),distribution_tv_max=maximum_tv,
                                distribution_tv_mean=tv_sum/positions,checked_positions=positions,
                                supervised_positions=int(sequence.loss_mask[:-1].sum()),sequence_sha256=sequence.invariant()))
    return dict(replay_layout='original sparse tree calls and KV operations' if teacher_trace is not None else 'full causal context',passed=True,hidden_atol=config.replay_hidden_atol,hidden_rtol=config.replay_hidden_rtol,
                distribution_tv_tolerance=config.replay_distribution_tv,examples=diagnostics,
                teacher_head_sha256=digest(head.state_dict()),distribution='full softmax, temperature=1, no truncation')


def relabel(target,sequences,*,zero_drift=False,teacher_trace=None):
    # With zero policy update, using the verified captured labels removes
    # replay/attention roundoff. The old gate still checked both label channels.
    updated=[]
    replayed_batch=teacher_trace.replay(target) if teacher_trace is not None and not zero_drift else None
    for index,s in enumerate(sequences):
        new=deepcopy(s)
        new.teacher_features=cpu_copy(s.teacher_features) if zero_drift else (replayed_batch[index] if replayed_batch is not None else teacher_features(target,s))
        if new.invariant()!=s.invariant():raise RuntimeError('Relabeling changed saved IDs/contexts/positions/masks')
        updated.append(new)
    return updated


def training_outputs(sequences,device):
    return dict(all_draft_input_states=[s.teacher_features.to(device).clone() for s in sequences],
                all_draft_input_ids=[s.draft_input_ids.to(device).clone() for s in sequences])


class TeacherTrace:
    """Compact teacher-only replay of original target calls and KV operations.

    Keep sparse tree-mask indices and short KV suffix selections, never full
    attention masks, KV tensors or full-V probabilities. Replaying these saved
    tokens is teacher forcing, with NO draft generation or target sampling.
    This also preserves BF16 attention shapes, which matter for real 3B outliers.
    """
    def __init__(self,repeats):
        self.repeats=repeats;self.events=[]

    def prefill(self,ids,positions,padding):
        self.events.append(dict(kind='prefill',ids=cpu_copy(ids),positions=cpu_copy(positions),
                                padding=[sorted(p) for p in padding]))

    def forward(self,ids,positions,past,padding,tree_indices):
        self.events.append(dict(kind='forward',ids=cpu_copy(ids),positions=cpu_copy(positions),past=past,
                                padding=[sorted(p) for p in padding],
                                tree_indices=cpu_copy(tree_indices).to(torch.int32) if torch.is_tensor(tree_indices) else torch.empty(0,3,dtype=torch.int32)))

    def chunk(self,indices,owners,padding,past,width):
        self.events.append(dict(kind='chunk',indices=cpu_copy(indices).to(torch.int32),owners=list(owners),
                                valid_slots=[[i for i in range(width) if past+i not in p] for p in padding]))

    def remove_row(self,index):self.events.append(dict(kind='remove_row',index=index))
    def select_rows(self,indices):self.events.append(dict(kind='select_rows',indices=cpu_copy(indices)))
    def crop(self,length):self.events.append(dict(kind='crop',length=length))
    def select_suffix(self,prefix,indices):
        self.events.append(dict(kind='select_suffix',prefix=prefix,indices=cpu_copy(indices).to(torch.int32)))

    @torch.no_grad()
    def replay(self,target):
        from helper.transformers_compat import DynamicCache
        base=base_model(target);device=base.device;dtype=base.dtype
        cache=DynamicCache();histories=None;hidden=None
        def forward(ids,mask,positions):
            # Same direct decoder API/arithmetic as fastgrpo_generate.model_forward.
            past=cache.get_seq_length()
            cache_position=torch.arange(past,past+ids.shape[1],device=device)
            states=base.model.embed_tokens(ids)
            rotary=base.model.rotary_emb(states,positions)
            for layer in base.model.layers[:base.model.config.num_hidden_layers]:
                result=layer(states,attention_mask=mask,position_ids=positions,past_key_value=cache,
                             output_attentions=False,use_cache=True,cache_position=cache_position,
                             position_embeddings=rotary)
                states=result[0]
            return base.model.norm(states)
        for event in self.events:
            kind=event['kind']
            if kind in ('prefill','forward'):
                ids=event['ids'].to(device);positions=event['positions'].to(device)
                b,q=ids.shape;past=cache.get_seq_length();minimum=torch.finfo(dtype).min
                if kind=='prefill':
                    if past:raise RuntimeError('Teacher trace must start from empty KV')
                    mask=torch.triu(torch.full((q,q),minimum,dtype=dtype,device=device),diagonal=1)[None,None].repeat(b,1,1,1)
                else:
                    if past!=event['past']:raise RuntimeError('Teacher replay KV/context length mismatch')
                    mask=torch.zeros((q,past+q),dtype=dtype,device=device)
                    mask[...,past+1:]=minimum;mask=mask[None,None].repeat(b,1,1,1)
                    indices=event['tree_indices'].to(device=device,dtype=torch.long)
                    if indices.numel():mask[indices[:,0],0,indices[:,1],indices[:,2]]=0
                for row,padding in enumerate(event['padding']):
                    if padding:mask[row,0,:,padding]=minimum
                with torch.autocast(device_type=device.type,dtype=dtype,enabled=device.type=='cuda'):
                    hidden=forward(ids,mask,positions)
                if kind=='prefill':
                    histories=[]
                    for row,padding in enumerate(event['padding']):
                        valid=[i for i in range(q) if i not in padding]
                        h=hidden[row,valid].detach().cpu().clone()
                        histories.extend([[h.clone()] for _ in range(self.repeats)])
                    if self.repeats>1:cache.batch_repeat_interleave(self.repeats)
            elif kind=='chunk':
                for row,owner in enumerate(event['owners']):
                    slots=event['valid_slots'][row]
                    selected=event['indices'][row,slots].to(device=device,dtype=torch.long)
                    histories[owner].append(hidden[row].index_select(0,selected).detach().cpu().clone())
                hidden=None
            elif kind=='select_rows':cache.batch_select_indices(event['indices'].to(device))
            elif kind=='remove_row':
                i=event['index']
                for layer in range(len(cache.key_cache)):
                    key,value=cache.key_cache[layer],cache.value_cache[layer]
                    cache.key_cache[layer]=torch.cat((key[:i],key[i+1:]),dim=0)
                    cache.value_cache[layer]=torch.cat((value[:i],value[i+1:]),dim=0)
            elif kind=='crop':cache.crop(event['length'])
            elif kind=='select_suffix':
                prefix=event['prefix'];indices=event['indices'].to(device=device,dtype=torch.long)
                keys=torch.stack(cache.key_cache);values=torch.stack(cache.value_cache)
                layers,b,heads,_,d=keys.shape
                index=indices[None,:,None,:,None].expand(layers,b,heads,-1,d)
                keys=torch.cat((keys[...,:prefix,:],keys[...,prefix:,:].gather(-2,index)),dim=-2)
                values=torch.cat((values[...,:prefix,:],values[...,prefix:,:].gather(-2,index)),dim=-2)
                for layer in range(layers):cache.key_cache[layer]=keys[layer];cache.value_cache[layer]=values[layer]
            else:raise ValueError(f'Unknown teacher trace operation {kind}')
        if histories is None:raise RuntimeError('Empty teacher trace')
        return [torch.cat(chunks) for chunks in histories]

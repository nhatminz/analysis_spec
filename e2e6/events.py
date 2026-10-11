"""Visited-path events only. No probability vectors or extra model forwards."""
import hashlib
import struct
import time
import torch

CENSORED = {'EOS','depth_cap','beam_cap','length_cap','prefill'}


def classify(y, candidates, retained, *, eos, depth, depth_limit, at_cap=False):
    if len(set(retained)) != len(retained):
        raise ValueError('Duplicate retained child tokens')
    if candidates is not None and not set(retained) <= set(candidates):
        raise ValueError('Retained children are not proposal candidates')
    if y == eos:return 'EOS',None
    if at_cap:return 'length_cap',None
    if candidates is None:
        if retained:raise ValueError('Children exist without a computed candidate context')
        return ('depth_cap' if depth >= depth_limit else 'beam_cap'),None
    if y in retained:return 'hit',False
    if y in candidates:return 'pruned',True
    return 'coverage',True


def canonical_ids(prompt_ids, repeats):
    return [(prompt, sample) for prompt in prompt_ids for sample in range(repeats)]


class VerificationLog:
    def __init__(self, step, rollout_id, prompt_ids, repeats, eos):
        self.step=step;self.rollout_id=rollout_id;self.mapping=canonical_ids(prompt_ids,repeats)
        self.eos=eos;self.events=[];self.rounds=[];self.tokens=[[] for _ in self.mapping]
        self.contexts=[];self.hashers=[];self.round_counts=[0]*len(self.mapping)

    def prefill(self, input_ids, attention_mask, sampled, repeats):
        prompts=input_ids.detach().cpu().tolist();masks=attention_mask.detach().cpu().tolist()
        sampled=sampled.detach().cpu().flatten().tolist()
        for i,(prompt,mask) in enumerate(zip(prompts,masks)):
            prefix=[v for v,m in zip(prompt,mask) if m]
            for sample in range(repeats):
                index=i*repeats+sample;h=hashlib.sha256()
                for token in prefix:h.update(struct.pack('<q',int(token)))
                self.contexts.append(list(prefix));self.hashers.append(h)
                self._append(index,sampled[i],dict(round_index=-1,tree_depth=-1,draft_top1=None,
                    draft_candidates=[],retained_children=[],reason='prefill',is_miss=None,
                    active_batch_size=len(self.mapping),draft_k=0,tree_budget=0,tree_nodes=0,accepted_length=0))

    def _append(self, canonical, y, metadata):
        if canonical < 0 or canonical >= len(self.mapping):raise ValueError('Invalid canonical response ID')
        if self.tokens[canonical] and self.tokens[canonical][-1]==self.eos:
            raise ValueError('Event emitted after EOS')
        prompt,sample=self.mapping[canonical];ctx=self.contexts[canonical]
        row=dict(checkpoint_step=self.step,rollout_id=self.rollout_id,response_id=canonical,
            prompt_id=prompt,group_id=prompt,sample_index=sample,token_position=len(self.tokens[canonical]),
            target_token=int(y),context2=ctx[-2:],context4=ctx[-4:],context8=ctx[-8:],
            full_context_sha256=self.hashers[canonical].hexdigest(),event_order=len(self.events),
            **metadata)
        self.events.append(row);self.tokens[canonical].append(int(y));ctx.append(int(y))
        self.hashers[canonical].update(struct.pack('<q',int(y)))

    def verification(self, *, trees, chosen, proposals, sampled, paths, emitted, canonical,
                     round_index, past_kv_len, active, draft_k, budget, depth_limit, capped=False):
        """Use the actual accepted path after EOS truncation, never rejected branches."""
        for row,original in enumerate(canonical):
            reverse={past_kv_len+1+j:index for j,index in enumerate(chosen[row])}
            contexts=proposals[row]
            current=-1
            accepted=len(emitted[row])
            if len(paths[row])!=accepted:raise ValueError('Accepted token/path count mismatch')
            if paths[row][0]!=past_kv_len:raise ValueError('Verification path did not start at root')
            for j,y in enumerate(emitted[row]):
                if j:
                    current=reverse[paths[row][j]]
                if current==-1:
                    child_indices=[index for index in chosen[row] if index < draft_k]
                    depth=0
                else:
                    child_indices=trees[row][current].child
                    depth=trees[row][current].depth+1
                retained=[int(trees[row][index].input_id) for index in child_indices]
                candidates=contexts.get(current)
                if candidates is not None and len(candidates)!=draft_k:
                    raise ValueError('Logged proposal context does not contain the full draft top-k')
                reason,miss=classify(int(y),candidates,retained,eos=self.eos,depth=depth,
                                    depth_limit=depth_limit,at_cap=capped and j==accepted-1)
                # An ordinary hit MUST lead to that exact retained child on the next path position.
                if reason=='hit':
                    matching=[idx for idx in child_indices if int(trees[row][idx].input_id)==int(y)]
                    if j+1>=accepted or paths[row][j+1]!=trees[row][matching[0]].target_index:
                        raise ValueError('Hit does not match generator accepted path')
                elif j+1<accepted:
                    raise ValueError('Generator continued after a miss or censored position')
                sample_position=paths[row][j]-past_kv_len
                if int(sampled[row][sample_position])!=int(y):raise ValueError('Target-sample alignment failure')
                self._append(original,int(y),dict(round_index=round_index,tree_depth=depth,
                    draft_top1=candidates[0] if candidates is not None else None,
                    draft_candidates=candidates or [],retained_children=retained,reason=reason,is_miss=miss,
                    active_batch_size=active,draft_k=draft_k,tree_budget=budget,tree_nodes=len(chosen[row])+1,
                    accepted_length=accepted))
            self.round_counts[original]+=1
        self.rounds.append(dict(checkpoint_step=self.step,rollout_id=self.rollout_id,round_index=round_index,
            active_batch_size=active,draft_k=draft_k,tree_budget=budget,tree_nodes=sum(map(len,chosen))+len(canonical),
            accepted_length=sum(map(len,emitted)),active_response_rounds=len(canonical)))

    def validate(self, output):
        if self.tokens != output['generated_token_ids']:raise ValueError('Event reconstruction differs from emitted responses')
        if self.round_counts != output['response_verification_rounds']:raise ValueError('Canonical round counts disagree')
        if [len(row)-1 for row in self.tokens] != output['response_accepted_length_sum']:
            raise ValueError('Event reconstruction disagrees with acceptance counters')
        validate_reconstruction(self.events,output['generated_token_ids'],self.eos)
        return True


def validate_reconstruction(events, responses, eos):
    reconstructed=[[] for _ in responses]
    for row in sorted(events,key=lambda x:x['event_order']):
        index=int(row['response_id'])
        if row['token_position']!=len(reconstructed[index]):raise ValueError('Noncontiguous response positions')
        if reconstructed[index] and reconstructed[index][-1]==eos:raise ValueError('Tokens after EOS')
        if row['reason'] not in CENSORED|{'hit','coverage','pruned'}:raise ValueError('Unknown path label')
        if row['is_miss'] is not None:
            expected,_=classify(row['target_token'],row['draft_candidates'],row['retained_children'],
                                eos=eos,depth=row['tree_depth'],depth_limit=999)
            if expected!=row['reason']:raise ValueError('Inconsistent hit/miss classification')
        elif row['reason'] not in CENSORED:raise ValueError('Missing ordinary outcome')
        reconstructed[index].append(row['target_token'])
    if reconstructed!=responses:raise ValueError('Reconstructed responses differ')
    return True


class ReplayTiming:
    """Unsynchronized host wall intervals; CUDA stream intervals are separate.

    No per-phase synchronize is added. The caller synchronizes only at rollout
    boundaries. Existing baseline synchronizations are preserved unchanged.
    """
    def __init__(self):self.segments=[];self.current=None
    def phase(self, name, active):
        now=time.perf_counter();event=torch.cuda.Event(enable_timing=True);event.record()
        if self.current is not None:
            phase,size,start,begin=self.current
            self.segments.append(dict(phase=phase,active_batch_size=size,host_wall_s=now-start,
                                      begin=begin,end=event))
        self.current=(name,active,now,event)
    def finish(self):
        self.phase('finished',0);self.current=None
    def resolve(self):
        return [dict(phase=s['phase'],active_batch_size=s['active_batch_size'],host_wall_s=s['host_wall_s'],
                     cuda_stream_elapsed_s=s['begin'].elapsed_time(s['end'])/1000)
                for s in self.segments]

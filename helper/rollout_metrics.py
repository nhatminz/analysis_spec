"""Buffered, host-only DataLoader iteration telemetry (including skipped batches)."""
import atexit
import csv
import time
from pathlib import Path

OPD_FIELDS=('kl','selected_states','visited_states','frontier_states',
    'target_mass_in_draft_top16','target_compact_mass','active_rows_mean','active_rows_max',
    'sparse_rounds','dense_rounds','fused_rounds','gemm_rounds')
KV_FIELDS=('iter_host_syncs','iter_host_syncs_per_round','iter_kv_cache_bytes',
           'iter_kv_full_reallocations','iter_kv_full_history_copies','iter_kv_history_copy_bytes','iter_kv_rows_moved','iter_kv_pool_allocations')
ITER_TIME_FIELDS=('iter_wall_time_s','iter_generation_tokens_per_s','iter_end_to_end_tokens_per_s',
                  'iter_draft_train_time_s','iter_target_train_time_s')
DRAFT_UPDATE_FIELDS=('iter_draft_sparse_kl','iter_draft_sparse_tv',
                     'iter_draft_update_committed','iter_draft_updates_cumulative')
FIELDS=('global_iter','epoch','batch_iter','method','grpo_step','used_items','eligible_prompts','total_prompts',
    'iter_draft_feature_loss','iter_draft_distribution_loss','iter_draft_total_loss',
    'iter_aal','cumulative_aal','iter_generation_time_s','cumulative_generation_time_s','cumulative_wall_time_s',
    'iter_rollout_tokens','cumulative_rollout_tokens','iter_verification_rounds','cumulative_verification_rounds',
    'iter_acceptance_rate','cumulative_acceptance_rate')+tuple('iter_opd_'+s for s in OPD_FIELDS)+KV_FIELDS+ITER_TIME_FIELDS+DRAFT_UPDATE_FIELDS


class RolloutMetricsWriter:
    @staticmethod
    def capture(outputs):
        # Never retain rollout hidden states / KV / training tensors through
        # the subsequent optimizer step. These counters are ALREADY host data.
        names=('total_acc_length','total_decoded_token_num','total_time_cost',
               'total_accepted_draft_tokens','total_proposed_draft_tokens','response_generated_tokens')
        return {key:value for key,value in outputs.items() if key in names or
                (key.startswith('opd_') and isinstance(value,(int,float)))}

    def __init__(self,path,method,*,state=None,flush_interval=1):
        if flush_interval<1:raise ValueError('rollout flush interval must be positive')
        self.state=dict(state or {});self.method=method;self.pending=None
        self.flush_interval=flush_interval;self.unflushed=0
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        # Resume at the checkpoint's iterator position, removing future rows.
        if path.exists():
            with path.open(newline='') as f:
                reader=csv.DictReader(f);rows=list(reader);previous_fields=reader.fieldnames
            limit=self.state.get('global_iter',0)
            kept=[r for r in rows if int(r['global_iter'])<=limit]
            if len(kept)!=len(rows) or previous_fields!=list(FIELDS):
                backup=path.with_suffix('.pre_resume.csv')
                if not backup.exists():backup.write_bytes(path.read_bytes())
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=FIELDS);w.writeheader();w.writerows(kept)
        new=not path.exists() or path.stat().st_size==0
        self.stream=path.open('a',newline='',buffering=65536)
        self.writer=csv.DictWriter(self.stream,fieldnames=FIELDS)
        if new:self.writer.writeheader()
        atexit.register(self.close)

    def begin(self,epoch,batch_iter,total_prompts,used_items):
        if self.pending is not None:raise RuntimeError('unfinished DataLoader iteration')
        self.pending=dict(epoch=epoch,batch_iter=batch_iter,total_prompts=total_prompts,
                          used_before=used_items,started_at=time.perf_counter())

    @staticmethod
    def values(outputs):
        o=outputs or {}
        return dict(accepted=float(o.get('total_acc_length',0)),rounds=int(o.get('total_decoded_token_num',0)),
            tokens=sum(o.get('response_generated_tokens',[])),generation=float(o.get('total_time_cost',0)),
            accepted_draft=int(o.get('total_accepted_draft_tokens',0)),proposed=int(o.get('total_proposed_draft_tokens',0)))

    def next_state(self,outputs):
        state=dict(self.state)
        if self.pending is not None:
            for k,v in self.values(outputs).items():state[k]=state.get(k,0)+v
            state['global_iter']=state.get('global_iter',0)+1
        return state

    def finish(self,outputs,*,grpo_step,used_items,wall_time_s,draft_updates_cumulative=0):
        if self.pending is None:return
        # Monotonic host wall time: only this DataLoader iteration, excluding
        # startup and earlier iterations. No event query or stream wait here.
        iter_wall_time_s=max(0.,time.perf_counter()-self.pending['started_at'])
        self.state=self.next_state(outputs)
        meta=self.pending;self.pending=None;o=outputs or {};values=self.values(o)
        def ratio(n,d):return n/d if d else 0.
        row=dict(global_iter=self.state['global_iter'],epoch=meta['epoch'],batch_iter=meta['batch_iter'],
            method=self.method,grpo_step=grpo_step,used_items=used_items,
            eligible_prompts=used_items-meta['used_before'],total_prompts=meta['total_prompts'],
            iter_aal=ratio(values['accepted'],values['rounds']),cumulative_aal=ratio(self.state['accepted'],self.state['rounds']),
            iter_generation_time_s=values['generation'],cumulative_generation_time_s=self.state['generation'],
            cumulative_wall_time_s=wall_time_s,iter_rollout_tokens=values['tokens'],cumulative_rollout_tokens=self.state['tokens'],
            iter_verification_rounds=values['rounds'],cumulative_verification_rounds=self.state['rounds'],
            iter_acceptance_rate=ratio(values['accepted_draft'],values['proposed']),
            cumulative_acceptance_rate=ratio(self.state['accepted_draft'],self.state['proposed']))
        row.update(iter_wall_time_s=iter_wall_time_s,
            iter_generation_tokens_per_s=ratio(values['tokens'],values['generation']),
            iter_end_to_end_tokens_per_s=ratio(values['tokens'],iter_wall_time_s))
        # These values must already be Python scalars. Never coerce a tensor
        # (including a scalar tensor) just to expose another CSV metric.
        for field in ('iter_draft_train_time_s','iter_target_train_time_s'):
            value=o.get(field,0.)
            row[field]=value if isinstance(value,(int,float)) else ''
        for field in ('iter_draft_sparse_kl','iter_draft_sparse_tv'):
            value=o.get(field)
            row[field]=value if isinstance(value,(int,float)) else ''
        value=o.get('iter_draft_update_committed',False)
        row['iter_draft_update_committed']=value if isinstance(value,(bool,int)) else ''
        row['iter_draft_updates_cumulative']=(draft_updates_cumulative
            if isinstance(draft_updates_cumulative,int) else '')
        for name in ('feature','distribution','total'):
            row['iter_draft_'+name+'_loss']=o.get('iter_draft_'+name+'_loss','')
        for field in OPD_FIELDS:row['iter_opd_'+field]=0.
        for field in KV_FIELDS:row[field]=''
        if self.method in ('fastgrpo','opd_reflex'):
            row['iter_host_syncs']=o.get('opd_host_syncs',0)
            row['iter_host_syncs_per_round']=o.get('opd_host_syncs_per_round',0)
            row['iter_kv_cache_bytes']=sum(o.get('opd_'+side+'_kv_cache_bytes',0) for side in ('target','draft'))
            row['iter_kv_rows_moved']=max(o.get('opd_'+side+'_kv_rows_moved',0) for side in ('target','draft'))
            row['iter_kv_pool_allocations']=sum(o.get('opd_'+side+'_pool_allocations',0) for side in ('target','draft'))
            for field,name in (('iter_kv_full_reallocations','full_kv_reallocations'),('iter_kv_full_history_copies','full_history_copies'),('iter_kv_history_copy_bytes','full_history_copy_bytes')):
                row[field]=sum(o.get('opd_'+side+'_'+name,0) for side in ('target','draft'))
            for name in ('selected_states','visited_states','frontier_states','active_rows_max'):
                row['iter_opd_'+name]=o.get('opd_'+name,0.)
            for name,num,den in [('kl','kl_sum','state_weight'),('target_mass_in_draft_top16','draft_topk_target_mass_sum','selected_states'),
                    ('target_compact_mass','compact_mass_sum','selected_states'),('active_rows_mean','active_rows_sum','rounds')]:
                row['iter_opd_'+name]=ratio(o.get('opd_'+num,0.),o.get('opd_'+den,0.))
            if o.get('opd_nonfinite_kl_states',0):row['iter_opd_kl']=''
            for mode in ('sparse','dense','fused','gemm'):row['iter_opd_'+mode+'_rounds']=o.get('opd_proposal_mode_'+mode+'_rounds',0.)
        self.writer.writerow(row);self.unflushed+=1
        if self.unflushed>=self.flush_interval:self.flush()

    def flush(self):
        if not self.stream.closed:self.stream.flush()
        self.unflushed=0

    def close(self):
        if not self.stream.closed:self.flush();self.stream.close()

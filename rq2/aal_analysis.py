"""Five-bin AAL, paired cluster bootstrap, and figures from committed logs."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from e2e6.analysis import ClusterBootstrap
from e2e6.collect import validate_rollout
from motivation.state import atomic_json, file_hash
from reviewer_controls.plotting import pyplot, curve

BINS=['0–20%','20–40%','40–60%','60–80%','80–100%']


def bin_rounds(rounds,rollout,arm):
    """One ratio of sums per bin. No verification means no AAL measurement."""
    frame=pd.DataFrame(rounds)
    total=len(frame)
    if total:
        frame=frame.sort_values('round_index').reset_index(drop=True)
        if frame.round_index.tolist()!=list(range(total)):raise ValueError('Noncontiguous verification round indices')
        for key in ('accepted_length','active_response_rounds','active_batch_size'):
            values=frame[key].to_numpy(float)
            if not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any():
                raise ValueError(f'Invalid round counter: {key}')
        if (frame.active_response_rounds<=0).any() or (frame.accepted_length<frame.active_response_rounds).any():
            raise ValueError('A verified response emits at least its terminal verification token')
        if not (frame.active_batch_size==frame.active_response_rounds).all():raise ValueError('Active response counts differ')
        if (np.diff(frame.active_batch_size)>0).any():raise ValueError('Active response population grew within rollout')
        # Integer midpoint formula avoids floating boundary ambiguity.
        frame['bin_index']=np.minimum(4,((2*np.arange(total)+1)*5)//(2*total))
    result=[]
    for index,label in enumerate(BINS):
        group=frame[frame.bin_index==index] if total else frame
        accepted=int(group.accepted_length.sum()) if len(group) else 0
        active=int(group.active_response_rounds.sum()) if len(group) else 0
        result.append(dict(rollout_id=rollout,arm=arm,bin_index=index,progress_bin=label,
            progress_midpoint_pct=10+20*index,total_verification_rounds=total,
            verification_rounds=len(group),accepted_length=accepted,active_response_rounds=active,
            mean_active_responses=active/len(group) if len(group) else None,
            min_active_responses=int(group.active_batch_size.min()) if len(group) else None,
            max_active_responses=int(group.active_batch_size.max()) if len(group) else None,
            AAL=accepted/active if active else None,valid_measurement=active>0))
    return result


def paired_summary(frame,paired,samples=2000,seed=2026):
    """Equal rollout weights; OFF/ON share valid clusters and bootstrap draws."""
    boot=ClusterBootstrap(paired,samples,seed);summary=[]
    for index,label in enumerate(BINS):
        subset=frame[frame.bin_index==index]
        off=subset[subset.arm=='off'].set_index('rollout_id').reindex(paired)
        on=subset[subset.arm=='on'].set_index('rollout_id').reindex(paired)
        valid=np.isfinite(off.AAL.to_numpy(float))&np.isfinite(on.AAL.to_numpy(float))
        den=boot.weights@valid.astype(float)
        available=den>0
        count=int(valid.sum())
        record=dict(bin_index=index,progress_bin=label,progress_midpoint_pct=10+20*index,
            paired_rollouts_total=len(paired),paired_rollouts_valid=count,
            paired_rollouts_missing=[r for r,good in zip(paired,valid) if not good],
            bootstrap_valid_resamples=int(available.sum()),
            ci_status='estimated' if count>=2 and available.sum()>=20 else 'insufficient_rollout_clusters')
        draws={}
        for arm,table in (('off',off),('on',on)):
            values=table.AAL.to_numpy(float)
            values=np.where(valid,values,0.)
            draws[arm]=(boot.weights@values)[available]/den[available]
            record[f'{arm}_AAL_mean']=float(values.sum()/count) if count else None
            ci=boot.interval(draws[arm]) if record['ci_status']=='estimated' else [None,None]
            record[f'{arm}_ci95_low'],record[f'{arm}_ci95_high']=ci
            record[f'{arm}_rollouts_individually_valid']=int(np.isfinite(table.AAL.to_numpy(float)).sum())
            record[f'{arm}_verification_rounds']=int(table.verification_rounds.sum())
            record[f'{arm}_active_response_rounds']=int(table.active_response_rounds.sum())
            record[f'{arm}_accepted_length']=int(table.accepted_length.sum())
            active=table.mean_active_responses.to_numpy(float)
            record[f'{arm}_mean_active_responses']=float(active[valid].mean()) if count else None
        record['paired_ON_minus_OFF']=record['on_AAL_mean']-record['off_AAL_mean'] if count else None
        ci=boot.interval(draws['on']-draws['off']) if record['ci_status']=='estimated' else [None,None]
        record['paired_difference_ci95_low'],record['paired_difference_ci95_high']=ci
        summary.append(record)
    return summary


def plot(output,summary):
    plt=pyplot();x=[r['progress_midpoint_pct'] for r in summary]
    fig,ax=plt.subplots(figsize=(3.35,2.55))
    for arm,label,color in (('off','SPARK OFF','#0072B2'),('on','SPARK ON','#D55E00')):
        stats=[dict(estimate=r[f'{arm}_AAL_mean'],ci95=[r[f'{arm}_ci95_low'],r[f'{arm}_ci95_high']]) for r in summary]
        curve(ax,x,stats,label,color)
    ax.set(xlabel='Rollout progress (%)',ylabel='Average Acceptance Length',xlim=(0,100))
    ax.set_xticks(x,['0–20','20–40','40–60','60–80','80–100'])
    ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
    if not any(r['paired_rollouts_valid'] for r in summary):ax.text(.5,.5,'No valid verification measurements',transform=ax.transAxes,ha='center',fontsize=7)
    fig.tight_layout(pad=.5)
    save_figure(fig,output/'aal_during_rollout.pdf')
    save_figure(fig,output/'aal_during_rollout.png')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(3.35,2.45))
    for arm,label,color in (('off','SPARK OFF','#0072B2'),('on','SPARK ON','#D55E00')):
        y=[r[f'{arm}_mean_active_responses'] if r[f'{arm}_mean_active_responses'] is not None else np.nan for r in summary]
        ax.plot(x,y,'o-',label=label,color=color,markersize=3,linewidth=1.)
    ax.set(xlabel='Rollout progress (%)',ylabel='Mean active responses',xlim=(0,100))
    ax.set_xticks(x,['0–20','20–40','40–60','60–80','80–100']);ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,fontsize=8);fig.tight_layout(pad=.5)
    save_figure(fig,output/'active_responses.pdf');plt.close(fig)


def save_figure(fig,path):
    temporary=path.with_name(path.name+'.tmp')
    options=dict(format=path.suffix[1:])
    if path.suffix=='.pdf':options['metadata']={'CreationDate':None,'ModDate':None}
    else:options['dpi']=250
    fig.savefig(temporary,**options)
    temporary.replace(path)


def analyze(output,samples=2000,seed=2026,smoke=False):
    started=time.perf_counter();output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    root=output/('smoke' if smoke else 'collection')
    ids={arm:{int(p.parent.name.split('_')[-1]) for p in (root/arm).glob('rollout_*/complete.json')} for arm in ('off','on')}
    paired=sorted(ids['off']&ids['on'])
    if not paired:raise ValueError('No committed OFF/ON pairs; no AAL results fabricated')
    binned=[];evidence=[];input_hashes={}
    for rollout in paired:
        paired_identity=None
        for arm in ('off','on'):
            directory=root/arm/f'rollout_{rollout:04d}'
            validate_rollout(directory,manifest['identity_sha256'])
            commit=json.loads((directory/'complete.json').read_text())
            if commit['arm']!=arm or commit['rollout_id']!=rollout or commit['checkpoint_step']!=manifest['config']['checkpoint_step']:
                raise ValueError('AAL commit metadata mismatch')
            response=json.loads((directory/'responses.json').read_text())
            identity=(response['prompt_ids'],response['seed'],response['frozen_weights_sha256'],response['learned_A_sha256'])
            if paired_identity is not None and identity!=paired_identity:raise ValueError('OFF/ON prompts/seeds/frozen weights differ')
            paired_identity=identity
            if response['seed']!=manifest['seeds'][rollout]:raise ValueError('Rollout seed differs from manifest')
            if response['learned_A_sha256']!=manifest['checkpoint']['projector']['A_sha256']:raise ValueError('A differs from checkpoint provenance')
            diagnostic=json.loads((directory/'diagnostics.json').read_text())
            if (not diagnostic['feedback_valid'] or diagnostic['sharing_scope']!='per_prompt_group'
                or not diagnostic['B_reset_verified'] or not diagnostic['frozen_weights_verified']):
                raise ValueError('AAL rollout validation did not pass')
            rounds=pq.read_table(directory/'rounds.parquet').to_pylist()
            bins=bin_rounds(rounds,rollout,arm)
            total=sum(r['accepted_length'] for r in bins);active=sum(r['active_response_rounds'] for r in bins)
            counter=response['counters']
            if total!=counter['total_acc_length'] or active!=counter['total_decoded_token_num']:
                raise ValueError('Saved rounds disagree with generation AAL counters')
            if len(rounds)!=diagnostic['acceptance']['verification_rounds']:raise ValueError('Saved verification counts disagree')
            binned.extend(bins)
            evidence.append(dict(rollout_id=rollout,arm=arm,**diagnostic['acceptance'],
                nonfinite_probability_space_KL_states=diagnostic['opd_nonfinite_kl_states'],
                group_b_norms=diagnostic['group_b_norms'],group_updates=diagnostic['group_updates']))
            input_hashes[str(directory/'complete.json')]=file_hash(directory/'complete.json')
    frame=pd.DataFrame(binned);summary=paired_summary(frame,paired,samples,seed)
    destination=output/'smoke_analysis' if smoke else output
    destination.mkdir(parents=True,exist_ok=True)
    for name,records in (('aal_by_rollout.csv',binned),('aal_summary.csv',summary)):
        target=destination/name;tmp=target.with_suffix('.csv.tmp')
        pd.DataFrame(records).to_csv(tmp,index=False);tmp.replace(target)
    plot(destination,summary)
    validation=dict(format='spark_rq2_aal_validation_v1',status='validated_partial' if len(paired)<manifest['config']['num_rollouts'] else 'validated_complete',
        smoke_only=smoke,paired_rollouts=paired,unpaired_rollouts={arm:sorted(ids[arm]-set(paired)) for arm in ids},
        requested_rollouts=manifest['config']['num_rollouts'],bootstrap_samples=samples,bootstrap_seed=seed,
        checkpoint=manifest['checkpoint'],training_metadata=manifest['training_metadata'],
        manifest_sha256=file_hash(output/'manifest.json'),
        AAL_aggregation='Ratio of sums within bin; equal-weight mean of rollout AAL on paired valid bins',
        missing_policy='No imputation; bins with zero verification rounds have no AAL, excluded from paired bin statistics',
        NaN_Inf_policy='No nonfinite probabilities/features/B or AAL. Diagnostic probability-space KL may be nonfinite; recorded separately, no overall KL mean',
        paired_cluster_bootstrap=True,events_reconstructed=True,counters_match=True,
        execution={name:json.loads((output/name).read_text()) for name in
            ('runtime.json','smoke_execution.json','collection_execution.json') if (output/name).is_file()},
        artifacts={name:file_hash(destination/name) for name in ('aal_by_rollout.csv','aal_summary.csv',
            'aal_during_rollout.pdf','aal_during_rollout.png','active_responses.pdf')},
        evidence=evidence,summary=summary,input_commits=input_hashes,runtime_s=time.perf_counter()-started)
    atomic_json(destination/'validation.json',validation)
    return dict(output_dir=str(destination),paired_rollouts=len(paired),runtime_s=validation['runtime_s'])

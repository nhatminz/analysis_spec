"""Verification ratio-of-sums and paired prompt-cluster uncertainty."""
from pathlib import Path
import csv
import io
import json
import numpy as np
from motivation.state import atomic_bytes, atomic_json

CONDITIONS = ('stale/new','fresh/new','reflex_off/old','reflex_on/old','reflex_off/new','reflex_on/new')
EFFECTS = ('A1_delta_lag','A2_gain_old','A2_gain_new','A2_drift_interaction')


def aal(rows):
    numerator = sum(r['accepted_draft_length_sum'] for r in rows)
    denominator = sum(r['verification_rounds'] for r in rows)
    if denominator <= 0 or any(r['verification_rounds'] <= 0 for r in rows):
        raise ValueError('AAL requires a positive actual verification denominator for every response')
    return numerator / denominator


def effects(values):
    old = values['reflex_on/old'] - values['reflex_off/old']
    new = values['reflex_on/new'] - values['reflex_off/new']
    return dict(A1_delta_lag=values['fresh/new']-values['stale/new'], A2_gain_old=old,
                A2_gain_new=new, A2_drift_interaction=new-old)


def summarize(rows, step, n, seed=2026, samples=2000):
    groups={c:[r for r in rows if r['condition']==c] for c in CONDITIONS}
    if len(rows) != 6*n or set(r['condition'] for r in rows) != set(CONDITIONS):
        raise ValueError(f'Boundary requires exactly six conditions x {n} prompts = {6*n} responses')
    ids=[r['prompt_id'] for r in groups[CONDITIONS[0]]]
    if len(set(ids)) != n: raise ValueError('Duplicate prompt/sample IDs')
    ordered={}
    for c,g in groups.items():
        if len(g)!=n or len({r['prompt_id'] for r in g})!=n or {r['prompt_id'] for r in g}!=set(ids):
            raise ValueError('All six conditions must contain the same distinct prompt clusters once')
        if any(r.get('sample_index',0)!=0 or r['policy_step']!=step for r in g):
            raise ValueError('Exactly one response/prompt and a single policy step are required')
        byid={r['prompt_id']:r for r in g};ordered[c]=[byid[i] for i in ids]
    values={c:aal(g) for c,g in ordered.items()}
    metric=dict(policy_step=step, eval_prompts=n, eval_trajectories=6*n,
                evidence_label='confirmation_64' if n==64 else ('exploratory_16' if n==16 else 'smoke_nonpublishable'),
                **effects(values))
    for c,g in ordered.items():
        name=c.replace('/','_')
        metric[name+'_aal']=values[c]
        metric[name+'_accepted_length_sum']=sum(r['accepted_draft_length_sum'] for r in g)
        metric[name+'_verification_rounds']=sum(r['verification_rounds'] for r in g)
    arrays={c:np.array([[r['accepted_draft_length_sum'],r['verification_rounds']] for r in g],dtype=np.float64)
            for c,g in ordered.items()}
    rng=np.random.default_rng(seed);draws={name:[] for name in EFFECTS}
    for _ in range(samples):
        chosen=rng.integers(0,n,n)  # SAME prompt ID resampled across all conditions
        ratios={c:(a[chosen,0].sum()/a[chosen,1].sum()) for c,a in arrays.items()}
        for name,value in effects(ratios).items():draws[name].append(value)
    for name in EFFECTS:
        lo,hi=np.quantile(draws[name],[.025,.975])
        metric[name+'_ci_low']=float(lo);metric[name+'_ci_high']=float(hi)
    return metric


def rebuild_exports(output_dir, committed_steps):
    root=Path(output_dir);responses=[];metrics=[]
    for step in sorted(committed_steps):
        journal=root/'boundaries'/f'step_{step}'/'results.json'
        payload=json.loads(journal.read_text())
        if payload['metrics']['policy_step']!=step:raise ValueError('Journal step mismatch')
        summarize(payload['per_response'],step,payload['metrics']['eval_prompts'],samples=1)
        atomic_json(journal.parent/'complete.json',dict(policy_step=step,durable_checkpoint=True))
        responses.extend(payload['per_response']);metrics.append(payload['metrics'])
    atomic_bytes(root/'per_response.jsonl', ''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in responses).encode())
    atomic_bytes(root/'step_metrics.jsonl', ''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in metrics).encode())
    if metrics:
        fields=sorted(set().union(*(r.keys() for r in metrics)))
        f=io.StringIO();w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(metrics)
        atomic_bytes(root/'step_metrics.csv',f.getvalue().encode())
    return metrics


def replot(output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(output_dir);checkpoint=root.parent/'checkpoints/latest.pt'
    # Results are committed only by a durable training checkpoint, never merely
    # by a result file surviving a crash before the target/draft save.
    if checkpoint.exists():
        import torch
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        metrics=rebuild_exports(root,saved['completed_analysis_steps'])
        del saved
    else: metrics=[json.loads(x) for x in (root/'step_metrics.jsonl').read_text().splitlines()]
    if not metrics:raise ValueError('No committed analysis boundaries to plot')
    plot_dir=root/'plots';plot_dir.mkdir(exist_ok=True)
    for name in ('A1_delta_lag','A2_gain_old','A2_gain_new','A2_drift_interaction'):
        fig,ax=plt.subplots(figsize=(6,3.5))
        for n,marker in ((16,'o'),(64,'s'),(2,'x'),(3,'x'),(4,'x')):
            selected=[r for r in metrics if r['eval_prompts']==n]
            if not selected:continue
            x=[r['policy_step'] for r in selected];y=[r[name] for r in selected]
            ax.errorbar(x,y,yerr=[[max(0.,r[name]-r[name+'_ci_low']) for r in selected],
                                   [max(0.,r[name+'_ci_high']-r[name]) for r in selected]],fmt=marker,
                        label=f'N={n}, '+('confirmation' if n==64 else ('exploratory' if n==16 else 'smoke')))
        ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='Completed target optimizer steps',ylabel=name+' (length-capped AAL)')
        ax.legend();fig.tight_layout();fig.savefig(plot_dir/(name+'.png'),dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3.5))
    for n,marker in ((16,'o'),(64,'s'),(2,'x'),(3,'x'),(4,'x')):
        selected=[r for r in metrics if r['eval_prompts']==n]
        if selected:ax.scatter([r['teacher_policy_tv_full_softmax'] for r in selected],
                              [r['A2_drift_interaction'] for r in selected],marker=marker,label=f'N={n}')
    ax.set(xlabel='Target drift: full-softmax TV on fixed prompt-final contexts',ylabel='A2 drift interaction')
    ax.legend();fig.tight_layout();fig.savefig(plot_dir/'interaction_vs_drift.png',dpi=180);plt.close(fig)
    return metrics

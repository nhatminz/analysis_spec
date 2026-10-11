"""E2–E6 from shared Parquet ONLY; no model loading or forward passes."""
from collections import defaultdict
from pathlib import Path
import json
import math
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from motivation.state import atomic_json,atomic_bytes,digest,file_hash
from e2e6.collect import validate_rollout

BINS=('64','32–63','16–31','8–15','<8')


def active_bin(n):
    if n==64:return '64'
    if n>=32:return '32–63'
    if n>=16:return '16–31'
    if n>=8:return '8–15'
    return '<8'


def distance_bin(d):
    if d is None:return 'none'
    if d==1:return '1'
    if d<=4:return '2–4'
    if d<=16:return '5–16'
    if d<=64:return '17–64'
    return '>64'


def position_bin(n):return '0–15' if n<16 else ('16–63' if n<64 else ('64–255' if n<256 else '256+'))


def exposure_bin(n):return 0 if n==0 else int(math.log2(n))+1


def exclusive_source(history,response,group):
    if response in history:return 'same_response'
    if any(g==group for g in history.values()):return 'sibling_response'
    if history:return 'different_prompt'
    return 'none'


def annotate(events):
    """All events in round r query history BEFORE any round-r updates."""
    histories={kind:defaultdict(dict) for kind in ('token','pair','context2','context4','context8','full')}
    last_same={};seen=0;rows=[]
    response_events=defaultdict(int);group_events=defaultdict(int)
    response_y=defaultdict(int);group_y=defaultdict(int);batch_y=defaultdict(int)
    response_pair=defaultdict(int);group_pair=defaultdict(int);batch_pair=defaultdict(int)
    for round_index,group in events.groupby('round_index',sort=True):
        if round_index<0:continue
        pending=[];exposure_updates=[]
        for event in group.to_dict('records'):
            if event['is_miss'] is None or pd.isna(event['is_miss']):continue
            y=int(event['target_token']);response=int(event['response_id']);prompt=event['prompt_id']
            d=int(event['draft_top1']);out={k:event[k] for k in
                ('rollout_id','round_index','response_id','prompt_id','tree_depth','draft_k','token_position','reason')}
            out.update(y=y,d=d,is_miss=bool(event['is_miss']),active_bin=active_bin(event['active_batch_size']),
                       position_bin=position_bin(event['token_position']),round_bin=distance_bin(round_index+1),
                       exposure_bin=exposure_bin(seen))
            event_counts=dict(same=response_events[response],group=group_events[prompt],batch=seen)
            y_counts=dict(same=response_y[(response,y)],group=group_y[(prompt,y)],batch=batch_y[y])
            pair_counts=dict(same=response_pair[(response,d,y)],group=group_pair[(prompt,d,y)],batch=batch_pair[(d,y)])
            for counts in (event_counts,y_counts,pair_counts):
                counts['sibling']=counts['group']-counts['same']
                counts['different']=counts['batch']-counts['group']
            for scope in event_counts:
                out[scope+'_events_exposure']=exposure_bin(event_counts[scope])
                out[scope+'_y_exposure']=exposure_bin(y_counts[scope])
                out[scope+'_pair_exposure']=exposure_bin(pair_counts[scope])
            exposure_updates.append((response,prompt,d,y))
            keys=dict(token=y,pair=(d,y),full=(y,event['full_context_sha256']))
            for n in (2,4,8):
                context=list(event[f'context{n}']);keys[f'context{n}']=(y,tuple(context)) if len(context)==n else None
            for kind,key in keys.items():
                history=histories[kind].get(key,{}) if key is not None else {}
                out[kind+'_source']=exclusive_source(history,response,prompt)
                out[kind+'_same']=response in history
                out[kind+'_group']=any(g==prompt for g in history.values())
                out[kind+'_batch']=bool(history)
            last=last_same.get((response,y))
            out['distance']=distance_bin(round_index-last if last is not None else None)
            rows.append(out)
            if out['is_miss']:pending.append((response,prompt,y,keys))
        # Verification of the WHOLE batch completes at the same causal boundary.
        for response,prompt,y,keys in pending:
            for kind,key in keys.items():
                if key is not None:histories[kind][key][response]=prompt
            last_same[(response,y)]=int(round_index)
        seen+=sum(not pd.isna(v) for v in group['is_miss'])
        for response,prompt,d,y in exposure_updates:
            response_events[response]+=1;group_events[prompt]+=1
            response_y[(response,y)]+=1;group_y[(prompt,y)]+=1;batch_y[y]+=1
            response_pair[(response,d,y)]+=1;group_pair[(prompt,d,y)]+=1;batch_pair[(d,y)]+=1
    return pd.DataFrame(rows)


class ClusterBootstrap:
    def __init__(self,clusters,samples=2000,seed=2026):
        self.clusters=list(clusters);self.index={v:i for i,v in enumerate(self.clusters)};self.samples=samples
        n=len(clusters)
        self.weights=np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=samples) if n else np.zeros((samples,0))

    @staticmethod
    def interval(values):
        values=np.asarray(values);values=values[np.isfinite(values)]
        return np.quantile(values,[.025,.975]).tolist() if len(values) else [None,None]

    def ratio(self,frame,numerator,denominator):
        sums=np.zeros(len(self.clusters));counts=np.zeros(len(self.clusters))
        if len(frame):
            indices=frame['rollout_id'].map(self.index).to_numpy(dtype=int)
            np.add.at(sums,indices,np.asarray(numerator,dtype=float));np.add.at(counts,indices,np.asarray(denominator,dtype=float))
        total=counts.sum();num=sums.sum()
        den=self.weights@counts;valid=den>0
        draws=(self.weights@sums)[valid]/den[valid]
        enough_clusters=int((counts>0).sum())>=2
        return dict(estimate=float(num/total) if total else None,
                    ci95=self.interval(draws) if enough_clusters else [None,None],
                    numerator=float(num),denominator=float(total),clusters=int((counts>0).sum()),
                    ci_status='estimated' if enough_clusters else 'insufficient_rollout_clusters',
                    bootstrap_valid_resamples=int(valid.sum()))


def matched_comparison(frame,flag,boot):
    """Exact-token and exposure/round/position/tree-stratified common-support risks.

    Fixed empirical population weights; recompute BOTH conditional risks within
    every rollout-cluster bootstrap. Empty support is reported, never imputed.
    """
    empty=dict(exposed_risk=None,unexposed_risk=None,lift=None,lift_ci95=[None,None],
               exposed_ci95=[None,None],unexposed_ci95=[None,None],matched_events=0,common_strata=0,
               bootstrap_valid_resamples=0)
    if not len(frame):return empty
    strata=['y','round_bin','position_bin','tree_depth','draft_k','active_bin','exposure_bin']
    if 'pair' in flag:strata+=['d']
    if 'cross_exposure_bin' in frame:strata+=['cross_exposure_bin']
    if 'cross_y_exposure_bin' in frame:strata+=['cross_y_exposure_bin']
    if 'pair' in flag and 'cross_pair_exposure_bin' in frame:strata+=['cross_pair_exposure_bin']
    for name in ('scope_events_exposure','scope_identity_exposure'):
        if name in frame:strata.append(name)
    keys=pd.MultiIndex.from_frame(frame[strata]);codes,unique=pd.factorize(keys)
    n=len(unique);r=len(boot.clusters)
    if not n:return empty
    indices=frame['rollout_id'].map(boot.index).to_numpy(int)
    exposed=frame[flag].to_numpy(bool);miss=frame['is_miss'].to_numpy(float)
    a=np.zeros((r,n));b=np.zeros((r,n));c=np.zeros((r,n));d=np.zeros((r,n))
    for matrix,values in ((a,miss*exposed),(b,exposed),(c,miss*~exposed),(d,~exposed)):
        np.add.at(matrix,(indices,codes),values)
    common=(b.sum(0)>0)&(d.sum(0)>0)
    if not common.any():return empty
    a,b,c,d=[v[:,common] for v in (a,b,c,d)]
    weights=b.sum(0)+d.sum(0);weights/=weights.sum()
    p=float(((a.sum(0)/b.sum(0))*weights).sum());q=float(((c.sum(0)/d.sum(0))*weights).sum())
    risks=[];controls=[];lifts=[]
    for start in range(0,len(boot.weights),100):
        w=boot.weights[start:start+100];aa=w@a;bb=w@b;cc=w@c;dd=w@d
        valid=(bb>0)&(dd>0);normal=(valid*weights).sum(1)
        pp=(np.divide(aa,bb,out=np.zeros_like(aa),where=bb>0)*weights*valid).sum(1)
        qq=(np.divide(cc,dd,out=np.zeros_like(cc),where=dd>0)*weights*valid).sum(1)
        pp=np.divide(pp,normal,out=np.full_like(pp,np.nan),where=normal>0)
        qq=np.divide(qq,normal,out=np.full_like(qq,np.nan),where=normal>0)
        risks.extend(pp);controls.extend(qq)
        lifts.extend(np.divide(pp,qq,out=np.full_like(pp,np.nan),where=qq>0))
    enough_clusters=int(((b+d).sum(1)>0).sum())>=2
    return dict(exposed_risk=p,unexposed_risk=q,lift=p/q if q>0 else None,
                lift_ci95=boot.interval(lifts) if enough_clusters else [None,None],
                exposed_ci95=boot.interval(risks) if enough_clusters else [None,None],
                unexposed_ci95=boot.interval(controls) if enough_clusters else [None,None],
                matched_events=int((b+d).sum()),common_strata=int(common.sum()),
                bootstrap_valid_resamples=int(np.isfinite(lifts).sum()))


def comparison(frame,flag,label,boot):
    frame=frame.copy()
    scope=('same' if flag.endswith('_same') or label.startswith('round_distance_') else 'batch')
    scope={'same_response':'same','sibling_response':'sibling','different_prompt':'different'}.get(label,scope)
    identity='pair' if 'pair' in flag else 'y'
    if scope+'_events_exposure' in frame:
        frame['scope_events_exposure']=frame[scope+'_events_exposure']
        frame['scope_identity_exposure']=frame[scope+'_'+identity+'_exposure']
    exp=frame[flag].astype(bool) if len(frame) else pd.Series([],dtype=bool)
    success=frame['is_miss'].astype(float) if len(frame) else np.array([])
    return dict(condition=label,baseline=boot.ratio(frame,success,np.ones(len(frame))),
        exposed=boot.ratio(frame,success*np.asarray(exp),exp),
        unexposed=boot.ratio(frame,success*~np.asarray(exp),~np.asarray(exp)),
        matched=matched_comparison(frame,flag,boot))


def add_controls(frames):
    """Disjoint donor/evaluation rollout halves; one donor per recipient, no reuse.

    Controls look only at donor rounds strictly BEFORE the recipient round index.
    Independent prompts are enforced by the collection plan.
    """
    ids=sorted(frames);half=len(ids)//2;results=[]
    for donor,recipient in zip(ids[:half],ids[half:2*half]):
        source=frames[donor];target=frames[recipient].copy()
        if not len(target):continue
        first={}
        for event in source.to_dict('records'):
            if event['is_miss']:
                for key,value in (('token',event['y']),('pair',(event['d'],event['y']))):
                    first[(key,value)]=min(first.get((key,value),math.inf),event['round_index'])
        prefix=source.groupby('round_index').size().sort_index().cumsum() if len(source) else pd.Series([],dtype=int)
        token_prefix={};pair_prefix={}
        if len(source):
            for y,occurrences in source.groupby('y'):
                token_prefix[y]=occurrences.groupby('round_index').size().sort_index().cumsum()
            for pair,occurrences in source.groupby(['d','y']):
                pair_prefix[pair]=occurrences.groupby('round_index').size().sort_index().cumsum()
        target['cross_prior']=[first.get(('token',y),math.inf)<r for y,r in zip(target['y'],target['round_index'])]
        target['cross_pair_prior']=[first.get(('pair',(d,y)),math.inf)<r for d,y,r in zip(target['d'],target['y'],target['round_index'])]
        target['cross_exposure_bin']=[exposure_bin(int(prefix[prefix.index<r].iloc[-1]) if (prefix.index<r).any() else 0)
                                      for r in target['round_index']]
        target['cross_y_exposure_bin']=[exposure_bin(int(token_prefix[y][token_prefix[y].index<r].iloc[-1])
            if y in token_prefix and (token_prefix[y].index<r).any() else 0) for y,r in zip(target['y'],target['round_index'])]
        target['cross_pair_exposure_bin']=[exposure_bin(int(pair_prefix[(d,y)][pair_prefix[(d,y)].index<r].iloc[-1])
            if (d,y) in pair_prefix and (pair_prefix[(d,y)].index<r).any() else 0)
            for d,y,r in zip(target['d'],target['y'],target['round_index'])]
        results.append(target)
    return pd.concat(results,ignore_index=True) if results else pd.DataFrame(),half


def tables(frames,rounds,timings,clusters,bootstrap_samples,seed):
    frame=pd.concat([v for v in frames.values() if len(v)],ignore_index=True) if any(len(v) for v in frames.values()) else pd.DataFrame()
    boot=ClusterBootstrap(clusters,bootstrap_samples,seed)
    e2=[];e3=[];e4=[];e5=[];e6=[]
    if len(frame):
        e2=[comparison(frame,flag,label,boot) for flag,label in
            [('token_same','same_response_token'),('token_batch','within_rollout_token'),('pair_same','same_response_confusion_pair')]]
        for distance in ('1','2–4','5–16','17–64','>64'):
            subset=frame[(frame['distance']==distance)|(frame['distance']=='none')].copy()
            subset['recurrence_at_distance']=subset['distance']==distance
            e2.append(comparison(subset,'recurrence_at_distance','round_distance_'+distance,boot))
        cross,pairs=add_controls(frames)
        if len(cross):
            cross_boot=ClusterBootstrap(sorted(cross['rollout_id'].unique()),bootstrap_samples,seed)
            for flag,label in [('token_batch','within_rollout_paired_control_population'),('cross_prior','cross_rollout_token'),
                               ('cross_pair_prior','cross_rollout_confusion_pair')]:
                row=comparison(cross,flag,label,cross_boot);row['independent_donor_recipient_pairs']=pairs;e2.append(row)
        for kind in ('token','context2','context4','context8','full'):
            for source in ('same_response','sibling_response','different_prompt'):
                subset=frame[frame[kind+'_source'].isin((source,'none'))].copy()
                subset['source_exposed']=subset[kind+'_source']==source
                row=comparison(subset,'source_exposed',source,boot);row['context']=kind;e3.append(row)
        for scope in ('same','group','batch'):
            for kind in ('token','pair','context2','context4','context8','full'):
                misses=frame[frame['is_miss']]
                for grouping,column in [('overall',None),('round_position','round_bin'),('response_position','position_bin'),
                                        ('prior_target_token_exposure',scope+'_y_exposure')]:
                    groups=[('all',misses)] if column is None else misses.groupby(column,sort=True)
                    for value,subset in groups:
                        stat=boot.ratio(subset,subset[kind+'_'+scope].astype(float),np.ones(len(subset)))
                        e6.append(dict(scope=scope,matching=kind,grouping=grouping,stratum=str(value),availability=stat))
    # Use raw rounds for AAL and proposal cost; ordinary events for miss denominators.
    for label in BINS:
        subset=frame[frame['active_bin']==label] if len(frame) else frame
        rr=rounds[rounds['active_batch_size'].map(active_bin)==label] if len(rounds) else rounds
        segments=[]
        for rollout,timing in timings.items():
            for segment in timing['segments']:
                if segment['phase'] in ('proposal','verification') and active_bin(segment['active_batch_size'])==label:
                    segments.append(dict(rollout_id=rollout,wall=segment['host_wall_s'],cuda=segment['cuda_stream_elapsed_s']))
        wall_by={r:sum(x['wall'] for x in segments if x['rollout_id']==r) for r in clusters}
        time_frame=pd.DataFrame([dict(rollout_id=r,wall=wall_by[r],total=timings[r]['wall_s']) for r in clusters])
        miss=subset['is_miss'].astype(float) if len(subset) else np.array([])
        ordinary=np.ones(len(subset));misses=subset[subset['is_miss']] if len(subset) else subset
        row=dict(active_bin=label,wall_clock_share=boot.ratio(time_frame,time_frame['wall'],time_frame['total']),
            host_wall_s=sum(wall_by.values()),cuda_stream_elapsed_s=sum(x['cuda'] for x in segments),
            miss_rate=boot.ratio(subset,miss,ordinary),
            aal=boot.ratio(rr,rr['accepted_length'] if len(rr) else [],rr['active_response_rounds'] if len(rr) else []),
            recurrence_share=boot.ratio(misses,misses['token_batch'].astype(float) if len(misses) else [],np.ones(len(misses))),
            mean_proposal_budget=boot.ratio(rr,rr['tree_budget'] if len(rr) else [],np.ones(len(rr))),
            nodes_per_response=boot.ratio(rr,rr['tree_nodes'] if len(rr) else [],rr['active_response_rounds'] if len(rr) else []),
            timing_attribution='measured host wall intervals; asynchronous GPU carry-over can cross bins')
        for reason in ('coverage','pruned'):
            row[reason+'_share']=boot.ratio(misses,(misses['reason']==reason).astype(float) if len(misses) else [],np.ones(len(misses)))
        e4.append(row)
    if len(frame):
        groupings=[('overall',None),('tree_depth','tree_depth'),('active_batch_size','active_bin'),('draft_k','draft_k')]
        for kind,column in groupings:
            groups=[('all',frame)] if column is None else frame.groupby(column,sort=True)
            for value,subset in groups:
                misses=subset[subset['is_miss']]
                row=dict(grouping=kind,value=str(value),ordinary_events=len(subset),misses=len(misses),
                    miss_rate=boot.ratio(subset,subset['is_miss'].astype(float),np.ones(len(subset))))
                for reason in ('coverage','pruned'):
                    row[reason]=boot.ratio(misses,(misses['reason']==reason).astype(float),np.ones(len(misses)))
                e5.append(row)
    return dict(E2=e2,E3=e3,E4=e4,E5=e5,E6=e6,ordinary_events=len(frame),
        clusters=len(clusters),bootstrap_resamples=bootstrap_samples,
        token_baseline=frame.groupby('y')['is_miss'].agg(['sum','count']).reset_index().to_dict('records') if len(frame) else [])


def flatten(row,prefix=''):
    result={}
    for key,value in row.items():
        name=prefix+key
        if isinstance(value,dict):result.update(flatten(value,name+'_'))
        elif isinstance(value,list):
            if len(value)==2 and all(v is None or isinstance(v,(int,float)) for v in value):
                result[name+'_low']=value[0];result[name+'_high']=value[1]
            else:result[name]=json.dumps(value)
        else:result[name]=value
    return result


def save_csv(path,rows):
    stream=pd.DataFrame([flatten(row) for row in rows]).to_csv(index=False)
    atomic_bytes(path,stream.encode())


def figures(destination,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import textwrap
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    specifications=[('recurrence.pdf','E2','condition','exposed','P(miss | prior feedback)'),
        ('response_sharing.pdf','E3','condition','exposed','Conditional miss probability'),
        ('long_tail.pdf','E4','active_bin','wall_clock_share','Share of replay wall time'),
        ('miss_types.pdf','E5','value','coverage','Fraction among ordinary misses')]
    for filename,table,xkey,statkey,ylabel in specifications:
        fig,ax=plt.subplots(figsize=(6.8,3.6));has_data=False
        for color_index,(step,result) in enumerate(sorted(summary.items(),key=lambda item:int(item[0]))):
            rows=result[table]
            if table=='E3':rows=[r for r in rows if r['context']=='token']
            if table=='E5':rows=[r for r in rows if r['grouping']=='tree_depth']
            for metric in (('coverage','pruned') if table=='E5' else (statkey,)):
                valid=[row for row in rows if row[metric]['estimate'] is not None]
                if not valid:continue
                names={'same_response_token':'Same response: token','within_rollout_token':'Within rollout: token',
                    'same_response_confusion_pair':'Same response: (d, y)',
                    'within_rollout_paired_control_population':'Within rollout: control population',
                    'cross_rollout_token':'Independent rollout: token',
                    'cross_rollout_confusion_pair':'Independent rollout: (d, y)'}
                labels=['\n'.join(textwrap.wrap(names.get(r[xkey],str(r[xkey]).replace('_',' ')),23)) for r in valid]
                values=[r[metric]['estimate'] for r in valid]
                lower=[max(0,v-r[metric]['ci95'][0]) if r[metric]['ci95'][0] is not None else np.nan for v,r in zip(values,valid)]
                upper=[max(0,r[metric]['ci95'][1]-v) if r[metric]['ci95'][1] is not None else np.nan for v,r in zip(values,valid)]
                label=f'Step {step}'+(f' · {metric}' if table=='E5' else '')
                ax.errorbar(labels,values,yerr=[lower,upper],marker='o',capsize=2,label=label,
                            linestyle='--' if metric=='pruned' else '-',color=plt.get_cmap('tab10')(color_index%10))
                has_data=True
        if not has_data:ax.text(.5,.5,'No valid measured observations',transform=ax.transAxes,ha='center')
        else:ax.legend(frameon=False,ncol=2 if table=='E5' else 1,fontsize=7)
        ax.set_ylabel(ylabel);ax.tick_params(axis='x',labelrotation=35);ax.grid(axis='y',alpha=.2)
        ax.set_ylim(-.03,1.03)
        if table=='E4':ax.set_xlabel('Active responses (proposal/verification phases)')
        elif table=='E5':ax.set_xlabel('Observed proposal-context depth')
        tmp=destination/(filename+'.tmp');fig.savefig(tmp,format='pdf');plt.close(fig);os.replace(tmp,destination/filename)


def analyze(root,steps,bootstrap_samples=2000,seed=2026):
    root=Path(root);destination=root/'analysis_e2_e6';destination.mkdir(parents=True,exist_ok=True)
    summary={};all_tables={name:[] for name in ('E2','E3','E4','E5','E6')};inputs={}
    for step in sorted(set(steps)):
        path=root/'collection'/f'step_{step:06d}'
        committed=[p.parent for p in sorted(path.glob('rollout_*/complete.json'))]
        if not committed:continue
        for folder in committed:
            validate_rollout(folder)
            inputs[str(folder.relative_to(root))]=file_hash(folder/'complete.json')
        cache=destination/'steps'/f'step_{step:06d}.json'
        identity=digest(dict(commits={str(p):file_hash(p/'complete.json') for p in committed},
                             step_manifest=file_hash(path/'manifest.json'),bootstrap_samples=bootstrap_samples,
                             seed=seed+step,source=file_hash(Path(__file__))))
        if cache.exists() and json.loads(cache.read_text())['identity_sha256']==identity:
            saved=json.loads(cache.read_text());result=saved['result']
            if digest(result)!=saved['result_sha256']:raise ValueError('Offline analysis cache integrity failed')
            summary[str(step)]=result
            for name in all_tables:all_tables[name].extend(dict(checkpoint_step=step,**row) for row in result[name])
            save_csv(destination/f'E2_token_baseline_step_{step}.csv',result.pop('token_baseline'))
            continue
        frames={};timings={};round_frames=[];censored=defaultdict(int)
        for folder in committed:
            validate_rollout(folder)
            commit=json.loads((folder/'complete.json').read_text());rollout=commit['rollout_id']
            events=pq.read_table(folder/'events.parquet').to_pandas()
            for reason,count in events['reason'].value_counts().items():censored[reason]+=int(count)
            frames[rollout]=annotate(events);round_frames.append(pq.read_table(folder/'rounds.parquet').to_pandas())
            timings[rollout]=json.loads((folder/'timing.json').read_text())
            inputs[str(folder.relative_to(root))]=file_hash(folder/'complete.json')
        rounds=pd.concat(round_frames,ignore_index=True)
        result=tables(frames,rounds,timings,sorted(frames),bootstrap_samples,seed+step)
        result['all_event_labels']=dict(censored)
        result['timing_note']='Logging runtime excluded. Host wall phase shares have asynchronous boundary attribution; CUDA stream intervals are separate and are not kernel-only times.'
        result['replay_wall_s']=sum(t['wall_s'] for t in timings.values())
        result['logging_wall_s']=sum(t['logging_wall_s'] for t in timings.values())
        atomic_json(cache,dict(identity_sha256=identity,result_sha256=digest(result),result=result))
        summary[str(step)]=result
        for name in all_tables:
            all_tables[name].extend(dict(checkpoint_step=step,**row) for row in result[name])
        save_csv(destination/f'E2_token_baseline_step_{step}.csv',result.pop('token_baseline'))
    for name,rows in all_tables.items():
        save_csv(destination/f'{name}.csv',rows);atomic_json(destination/f'{name}.json',rows)
    atomic_json(destination/'summary.json',summary)
    figures(destination,summary)
    report=['# E2–E6 measured motivation analysis','',
        'Pure frozen FastGRPO. Each checkpoint uses one shared verification event log; offline analysis performs zero model forward passes.',
        'Ordinary miss denominators exclude prefill, EOS, depth/beam caps and length censoring.',
        'Causal feedback includes only strictly earlier verification rounds; same-round observations are simultaneous.',
        'E3 sources are mutually exclusive: same response, otherwise sibling response, otherwise different prompt.',
        'E6 measures information availability, not successful correction.',
        'Cross-rollout controls use disjoint donor/recipient rollout halves with one donor per recipient; prompts are disjoint across rollouts.',
        'Matched lifts condition on token ID, round/position bins, depth, draft k, active size, total opportunities and prior actual-token/confusion-pair exposure bins within the relevant scope. Empty common support yields null.',
        'Fewer than two contributing independent rollout clusters yields no confidence interval. E6 includes round/response-position and prior-token-exposure strata.',
        '95% intervals resample whole rollouts (2,000 by default). Checkpoints are analyzed separately; repeated prompt responses remain inside their rollout cluster.',
        'Timing comes from event-disabled equivalent replay. Bin shares are CPU phase-wall intervals; asynchronous GPU carry-over can cross boundaries. CUDA stream elapsed intervals are reported separately and are not kernel-only timings.','']
    for step,result in summary.items():
        report += [f'## Checkpoint {step}',f'Valid ordinary events: {result["ordinary_events"]}; completed rollout clusters: {result["clusters"]}.',
                   f'Replay wall time: {result["replay_wall_s"]:.3f} s; logging runtime: {result["logging_wall_s"]:.3f} s (excluded from performance).']
        for row in result['E2']:
            matched=row['matched'];lift=matched['lift'];lo,hi=matched['lift_ci95']
            if lift is None or lo is None:
                report.append(f'- {row["condition"]}: insufficient common support or independent rollout clusters for a matched lift interval; {matched["matched_events"]} matched events.')
            else:
                finding='higher conditional miss risk' if lo>1 else ('lower conditional miss risk' if hi<1 else 'interval includes no lift')
                report.append(f'- {row["condition"]}: matched lift {lift:.3f}, 95% CI [{lo:.3f}, {hi:.3f}]; {finding}.')
        report += ['', 'These are observational associations, not proof that an adaptation would fix the misses.','']
    if not summary:report.append('No valid committed rollouts are available. No findings or synthetic results were generated.')
    atomic_bytes(destination/'REPORT.md',('\n'.join(report)+'\n').encode())
    atomic_json(destination/'manifest.json',dict(input_commit_sha256=inputs,checkpoints=list(summary),
        bootstrap_samples=bootstrap_samples,seed=seed,implementation_sha256=file_hash(Path(__file__)),
        collection_manifest_sha256=file_hash(root/'collection/manifest.json') if (root/'collection/manifest.json').exists() else None))
    atomic_json(destination/'complete.json',dict(manifest_sha256=file_hash(destination/'manifest.json'),
        files={p.name:file_hash(p) for p in destination.iterdir() if p.is_file() and p.name!='complete.json'}))
    return dict(checkpoints=list(summary),output_dir=str(destination))

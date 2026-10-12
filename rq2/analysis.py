"""RQ2 repeated misses only. Four arm/exposure cells share joint support."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from motivation.state import atomic_json, atomic_bytes, digest, file_hash
from e2e6.analysis import annotate, ClusterBootstrap, save_csv
from e2e6.collect import validate_rollout
from reviewer_controls.statistics import SAME_STRATA, DISTANCES, joint_match, inference


def comparison(off, on, boot):
    result=[]
    for distance in ('all',*DISTANCES):
        frames=[]
        for arm,frame in (('off',off),('on',on)):
            subset=frame.copy()
            if len(subset):
                if distance!='all':subset=subset[(~subset.token_same)|(subset.distance==distance)].copy()
                subset['cell']=np.where(subset.token_same,arm+'_prior',arm+'_no_prior')
            frames.append(subset)
        joint=pd.concat(frames,ignore_index=True)
        matched=joint_match(joint,['off_prior','off_no_prior','on_prior','on_no_prior'],boot,SAME_STRATA)
        change=matched['contrasts']['on_prior_minus_off_prior']
        conclusion=inference(change['risk_difference'],
            'SPARK ON has higher matched conditional recurrence probability.',
            'SPARK ON has lower matched conditional recurrence probability.',
            'The interval does not establish a change in conditional recurrence probability.')
        result.append(dict(distance=distance,matched=matched,
            recurrence_absolute_change=change['risk_difference'],recurrence_relative_change=change['relative_change'],
            conclusion=conclusion))
    return result


def analyze(destination, bootstrap_samples=2000, seed=2026):
    started=time.perf_counter();destination=Path(destination)
    manifest=json.loads((destination/'manifest.json').read_text())
    all_rows=[];summary={};inputs={}
    for step in sorted(manifest['checkpoints'],key=int):
        arm_frames={};arm_ids={};counts={};paired_prompts={}
        for arm in ('off','on'):
            folder=destination/'collection'/f'step_{int(step):06d}'/arm
            commits=sorted(folder.glob('rollout_*/complete.json'))
            arm_ids[arm]={int(json.loads(p.read_text())['rollout_id']) for p in commits}
        paired=sorted(arm_ids['off']&arm_ids['on'])
        if not paired:continue
        for arm in ('off','on'):
            frames=[]
            for rollout in paired:
                folder=destination/'collection'/f'step_{int(step):06d}'/arm/f'rollout_{rollout:04d}'
                validate_rollout(folder,manifest['identity_sha256'])
                commit=json.loads((folder/'complete.json').read_text())
                expected=manifest['checkpoints'][step]
                if commit['arm']!=arm or commit['checkpoint_identity']!=expected:
                    raise ValueError('Paired arm checkpoint/projector identity differs')
                response=json.loads((folder/'responses.json').read_text())
                if commit['projector_sha256']!=response['projector_sha256']:raise ValueError('Projector integrity differs')
                projector_file=destination/'projectors'/f'step_{int(step):06d}.json'
                projector_info=json.loads(projector_file.read_text())
                if projector_info['actual_A_sha256']!=response['projector_sha256']:raise ValueError('Projector differs from own-checkpoint provenance')
                inputs[str(projector_file)]=file_hash(projector_file)
                prompts=(response['prompt_ids'],response['seed'],response['initial_weights_sha256'],response['projector_sha256'])
                if rollout in paired_prompts and prompts!=paired_prompts[rollout]:raise ValueError('Arm prompts/seeds/initial weights differ')
                paired_prompts[rollout]=prompts
                raw=pq.read_table(folder/'events.parquet').to_pandas();frames.append(annotate(raw))
                inputs[str(folder/'complete.json')]=file_hash(folder/'complete.json')
            arm_frames[arm]=pd.concat(frames,ignore_index=True)
            counts[arm]=dict(paired_rollouts=len(paired),unpaired_rollouts=sorted(arm_ids[arm]-set(paired)),
                            ordinary_events=len(arm_frames[arm]))
        boot=ClusterBootstrap(paired,bootstrap_samples,seed+int(step))
        rows=comparison(arm_frames['off'],arm_frames['on'],boot)
        overall={arm:boot.ratio(frame,frame['is_miss'].astype(float),np.ones(len(frame))) if len(frame) else boot.ratio(frame,[],[])
                 for arm,frame in arm_frames.items()}
        summary[step]=dict(overall_miss_rate=overall,overall=rows[0],distances=rows[1:],counts=counts,
                           projector=manifest['checkpoints'][step]['projector'])
        all_rows.extend(dict(checkpoint_step=int(step),**row) for row in rows)
    if not summary:raise ValueError('No fully committed paired rollouts; no RQ2 findings generated')
    result=dict(checkpoints=summary,bootstrap_samples=bootstrap_samples,seed=seed,
                paired_bootstrap=True,event_pairing=False,input_commits=inputs,
                stratum_population='Joint common support across OFF prior/control and ON prior/control; pooled fixed weights',
                runtime_s=time.perf_counter()-started)
    atomic_json(destination/'rq2_repeated_misses.json',result)
    save_csv(destination/'rq2_repeated_misses.csv',all_rows)
    from reviewer_controls.plotting import plot_rq2
    plot_rq2(destination,result)
    report=['# RQ2: Repeated Draft Misses','',
        'Frozen target, draft and A within checkpoint; only fast correction B is updated in the ON arm. OFF uses the same SPARK generator with fast learning rate zero.',
        'Actual generation in both arms. Seed and prompt assignments are paired; individual token events and generated trajectories are not paired.',
        'Ordinary misses use pre-pruning proposal candidates and retained children, excluding prefill, EOS and depth/beam/length caps.',
        'History uses only strictly preceding rounds. Each distance compares recurrence in that bin against no-prior-miss events.',
        'Matching uses exact token, round/position bins, depth, draft k, active size, total exposure, response exposure and target-token exposure.',
        'All FOUR arm/exposure risks use identical joint common strata and fixed empirical event weights. Support and event composition changes are reported per cell.',
        '95% bootstrap intervals resample whole PAIRED rollout IDs; no one-to-one token event correspondence is assumed.',
        'A bootstrap draw missing any required cell/stratum is invalid; insufficient supported draws or clusters produce no confidence interval.',
        'Lower overall miss rate alone does not establish lower conditional recurrence.','']
    for step,row in summary.items():
        report += [f'## Checkpoint {step}',row['projector']['interpretation'],
                   f'Event composition/support: {row["counts"]}.',f'Overall ordinary miss sanity rates: {row["overall_miss_rate"]}.']
        for item in (row['overall'],*row['distances']):
            report.append(f'- Distance {item["distance"]}: ON–OFF conditional recurrence RD {item["recurrence_absolute_change"]}; relative change {item["recurrence_relative_change"]}. {item["conclusion"]}')
            for arm in ('off','on'):
                stat=item['matched'];contrast=stat['contrasts'][arm+'_prior_minus_'+arm+'_no_prior']
                report.append(f'  {arm.upper()}: prior={stat["risks"][arm+"_prior"]}; control={stat["risks"][arm+"_no_prior"]}; RD/RR={contrast}.')
            report.append(f'  Common-support coverage: {item["matched"]["support"]}.')
        report.append('')
    atomic_bytes(destination/'RQ2_REPEATED_MISSES_REPORT.md',('\n'.join(report)+'\n').encode())
    atomic_json(destination/'analysis_complete.json',dict(input_commits=inputs,runtime_s=time.perf_counter()-started,
        files={name:file_hash(destination/name) for name in ('rq2_repeated_misses.csv','rq2_repeated_misses.json',
            'RQ2_REPEATED_MISSES_REPORT.md','rq2_repeated_misses.pdf','rq2_repeated_misses.png')}))
    return dict(checkpoints=list(summary),output_dir=str(destination),runtime_s=time.perf_counter()-started)

"""Read-only E2/E3/E6 controls. This module never builds or evaluates a model."""
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from motivation.state import atomic_json, atomic_bytes, digest, file_hash
from e2e6.analysis import annotate, add_controls, ClusterBootstrap, save_csv
from e2e6.collect import validate_rollout
from reviewer_controls.statistics import BASE_STRATA, SAME_STRATA, DISTANCES, joint_match, inference, estimate


def output_directory(root, destination, default):
    root = Path(root).resolve()
    destination = Path(destination).resolve() if destination else root/default
    repository = Path(__file__).resolve().parents[1]
    if not destination.is_relative_to(repository):raise ValueError('Output must stay inside analysis_spec')
    if destination.parent != root or destination.name in ('collection','checkpoints','analysis_e2_e6','logs'):
        raise ValueError('Use a NEW direct child of the experiment root, separate from original artifacts')
    return destination


def read_collection(root, steps, allow_partial=False):
    """Validate immutable commits, checkpoint identity and disjoint prompt plan."""
    collection = Path(root)/'collection'
    if not (collection/'manifest.json').is_file():
        raise FileNotFoundError(f'Existing collection unavailable: {collection}; no collection will be generated')
    plan = json.loads((collection/'manifest.json').read_text())
    if plan.get('format') != 'e2_e6_shared_verification_v1':raise ValueError('Unsupported baseline collection schema')
    ids = plan['prompt_ids']; hashes = plan['question_hashes']
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):raise ValueError('Donor/recipient prompts are not independent')
    if (plan['batch_size'],plan['responses_per_prompt']) != (8,8) and not allow_partial:
        raise ValueError('Production controls require the original 8 x 8 collection')
    inputs = {str(collection/'manifest.json'):file_hash(collection/'manifest.json')}
    for name in ('train_manifest.json','external_checkpoint_manifest.json'):
        source=Path(root)/name
        if source.is_file():
            if plan.get('training_manifest_sha256')!=digest(json.loads(source.read_text())):
                raise ValueError('Original source manifest differs from the collected checkpoint plan')
            inputs[str(source)]=file_hash(source)
    loaded = {}; availability = {}
    for step in steps:
        folder = collection/f'step_{step:06d}'
        committed = sorted(folder.glob('rollout_*/complete.json'))
        availability[str(step)] = dict(committed_rollouts=len(committed),planned_rollouts=plan['eval_rollouts'])
        if len(committed) != plan['eval_rollouts'] and not allow_partial:
            raise ValueError(f'Step {step}: {len(committed)}/{plan["eval_rollouts"]} committed rollouts; --allow-partial is explicit')
        if not committed:continue
        meta = json.loads((folder/'manifest.json').read_text())
        if meta['shared_plan_sha256'] != digest(plan) or meta['checkpoint']['policy_step'] != step:
            raise ValueError('Checkpoint/shared-plan identity mismatch')
        if meta['identity_sha256'] != digest(dict(plan=plan,checkpoint=meta['checkpoint'])):
            raise ValueError('Frozen checkpoint identity digest differs from the step manifest')
        inputs[str(folder/'manifest.json')] = file_hash(folder/'manifest.json')
        frames = {}; raw_frames = {}; rounds = {}; used_prompts = set()
        for marker in tqdm(committed,desc=f'Validate baseline {step}',unit='rollout',dynamic_ncols=True):
            directory = marker.parent
            validate_rollout(directory,meta['identity_sha256'])
            commit = json.loads(marker.read_text()); rollout = int(commit['rollout_id'])
            if directory.name != f'rollout_{rollout:04d}' or rollout in frames:raise ValueError('Rollout identity mismatch')
            response = json.loads((directory/'responses.json').read_text())
            expected = ids[rollout*plan['batch_size']:(rollout+1)*plan['batch_size']]
            if response['prompt_ids'] != expected or not expected:raise ValueError('Rollout prompt assignment differs from plan')
            if used_prompts.intersection(expected):raise ValueError('Rollout prompt overlap')
            used_prompts.update(expected)
            raw = pq.read_table(directory/'events.parquet').to_pandas()
            rd = pq.read_table(directory/'rounds.parquet').to_pandas()
            if len(raw) and (not (raw.checkpoint_step==step).all() or not (raw.rollout_id==rollout).all()
                             or not set(raw.prompt_id).issubset(expected)):
                raise ValueError('Event metadata differs from frozen checkpoint/rollout')
            if len(rd) and (list(rd.round_index)!=list(range(len(rd))) or not (rd.rollout_id==rollout).all()):
                raise ValueError('Noncontiguous verification rounds')
            # Each response and verification round must be represented exactly.
            expected_counters = response['counters']
            ordinary_rounds = raw[raw.round_index>=0]
            n = len(response['tokens'])
            actual = ordinary_rounds.groupby('response_id').round_index.nunique().reindex(range(n),fill_value=0).tolist()
            if actual != expected_counters['response_verification_rounds']:raise ValueError('Round counters disagree with log')
            if sum(rd.accepted_length) != sum(expected_counters['response_accepted_length_sum']):raise ValueError('Round accepted-token totals differ')
            for path in directory.iterdir():
                if path.is_file():inputs[str(path)] = file_hash(path)
            annotated = annotate(raw)
            ordinary = raw[(raw.round_index>=0)&raw.is_miss.notna()].sort_values('round_index',kind='stable')
            if len(annotated):annotated['context4_eligible'] = ordinary.context4.map(len).to_numpy()==4
            frames[rollout] = annotated; raw_frames[rollout] = raw; rounds[rollout] = rd
        loaded[str(step)] = dict(frames=frames,raw=raw_frames,rounds=rounds,checkpoint=meta['checkpoint'])
    if not loaded:raise ValueError('No committed requested checkpoints; no findings were generated')
    return loaded, plan, inputs, availability


def paired_controls(frames):
    ids = sorted(frames); half = len(ids)//2
    for donor, recipient in zip(ids[:half],ids[half:2*half]):
        if len(frames[donor]) and len(frames[recipient]) and set(frames[donor].prompt_id)&set(frames[recipient].prompt_id):
            raise ValueError('Independent donor and recipient share prompts')
    control, _ = add_controls(frames)
    pairs = [dict(donor=d,recipient=r) for d,r in zip(ids[:half],ids[half:2*half])]
    return control, pairs, ids[2*half:]


def e2_controls(frame, boot):
    cells = ['current_response','independent_only','no_prior']
    rows = []
    for distance in ('all',*DISTANCES):
        subset = frame.copy()
        if len(subset):
            if distance != 'all':subset = subset[(~subset.token_same)|(subset.distance==distance)].copy()
            subset['cell'] = np.where(subset.token_same,'current_response',np.where(subset.cross_prior,'independent_only','no_prior'))
        stat = joint_match(subset,cells,boot,SAME_STRATA+['cross_exposure_bin','cross_y_exposure_bin'])
        difference = stat['contrasts']['current_response_minus_independent_only']['risk_difference']
        finding = inference(difference,'Within-response conditional risk is greater than the independent-only control.',
                            'Within-response conditional risk is lower than the independent-only control.',
                            'The interval does not establish a difference from the independent-only control.')
        rows.append(dict(distance=distance,**stat,conclusion=finding))
    return rows


def e3_controls(frame, boot):
    rows = []
    for definition in ('token','context4'):
        for source,scope in (('sibling_response','sibling'),('different_prompt','different')):
            subset = frame[frame[definition+'_source'].isin((source,'none'))].copy() if len(frame) else frame.copy()
            if definition=='context4' and 'context4_eligible' in subset:subset=subset[subset.context4_eligible].copy()
            if len(subset):subset['cell'] = np.where(subset[definition+'_source']==source,'exposed','unexposed')
            stat = joint_match(subset,['exposed','unexposed'],boot,
                               BASE_STRATA+[scope+'_events_exposure',scope+'_y_exposure'])
            finding = inference(stat['contrasts']['exposed_minus_unexposed']['risk_difference'],
                'Higher conditional miss risk than the matched unexposed control.',
                'Lower conditional miss risk than the matched unexposed control; sharing is not supported.',
                'No supported increase over the matched unexposed control (interval includes zero).')
            rows.append(dict(definition=definition,source=source,**stat,conclusion=finding))
    return rows


class PermutationHistory:
    """Fixed eligible positions; censored labels never enter or move in the null."""
    def __init__(self, raw):
        self.valid_mask = (raw.round_index>=0)&raw.is_miss.notna()
        frame = raw.loc[self.valid_mask].copy()
        self.miss = frame.is_miss.to_numpy(bool)
        self.rounds = frame.round_index.to_numpy(int)
        self.tokens = frame.target_token.to_numpy(int)
        self.token_codes, _ = pd.factorize(self.tokens,sort=True)
        self.token_order = np.argsort(self.token_codes,kind='stable')
        self.keys = []
        for columns in (['target_token','response_id'],['target_token','prompt_id'],['target_token']):
            codes, unique = pd.factorize(pd.MultiIndex.from_frame(frame[columns]),sort=True)
            self.keys.append((codes,len(unique)))

    def permute(self, rng):
        order = np.lexsort((rng.random(len(self.miss)),self.token_codes))
        shuffled = np.empty_like(self.miss)
        shuffled[order] = self.miss[self.token_order]
        return shuffled

    def counts(self, labels):
        labels = np.asarray(labels,dtype=bool)
        if labels.shape != self.miss.shape:raise ValueError('Permutation changed eligible positions')
        result = []
        for codes,size in self.keys:
            first = np.full(size,np.iinfo(np.int64).max,dtype=np.int64)
            np.minimum.at(first,codes[labels],self.rounds[labels])
            prior = first[codes] < self.rounds
            result.append(int((labels&prior).sum()))
        return result


def e6_frequency(raw_frames, boot, permutations, seed):
    if permutations<1:raise ValueError('Positive permutation count required')
    histories = [PermutationHistory(raw_frames[c]) for c in boot.clusters]
    denominators = np.array([h.miss.sum() for h in histories],dtype=float)
    observed = np.array([h.counts(h.miss) for h in histories],dtype=float).reshape(len(histories),3)
    null = np.zeros((permutations,len(histories),3))
    rng = np.random.default_rng(seed)
    for b in tqdm(range(permutations),desc='E6 token-frequency null',unit='permutation',dynamic_ncols=True):
        for r,h in enumerate(histories):null[b,r] = h.counts(h.permute(rng))
    rows = []; total = denominators.sum(); bootstrap_den = boot.weights@denominators
    enough = int((denominators>0).sum())>=2
    for s,scope in enumerate(('same_response','same_prompt_group','whole_batch')):
        value = observed[:,s].sum()/total if total else None
        null_values = null[:,:,s].sum(1)/total if total else np.array([])
        mean = float(null_values.mean()) if len(null_values) else None
        expected = null[:,:,s].mean(0)
        obs_draws = np.divide(boot.weights@observed[:,s],bootstrap_den,out=np.full(len(bootstrap_den),np.nan),where=bootstrap_den>0)
        delta_draws = np.divide(boot.weights@(observed[:,s]-expected),bootstrap_den,
                                out=np.full(len(bootstrap_den),np.nan),where=bootstrap_den>0)
        variable = bool(len(null_values) and np.ptp(null_values)>0)
        p = (1+int((np.abs(null_values-mean)>=abs(value-mean)-1e-15).sum()))/(permutations+1) if variable and permutations>=19 and enough else None
        rows.append(dict(scope=scope,observed=estimate(value,obs_draws,enough),null_mean=mean,
            null_interval95=np.quantile(null_values,[.025,.975]).tolist() if len(null_values) else [None,None],
            observed_minus_null=estimate(value-mean if total else None,delta_draws,enough),
            permutation_p_two_sided=p,permutation_p_definition='(1 + |Tperm-null_mean| >= |Tobs-null_mean| count)/(M+1)',
            p_status='estimated' if p is not None else 'insufficient_permutations_clusters_or_degenerate_null',
            permutations=permutations,eligible_events=sum(len(h.miss) for h in histories),
            censored_events=sum(len(raw_frames[c])-len(h.miss) for c,h in zip(boot.clusters,histories)),
            miss_events=int(total),rollout_clusters=len(histories),miss_clusters=int((denominators>0).sum()),
            interpretation='Token-frequency permutation NULL; not a measured SPARK result'))
    return rows


def e6_donor(frame, boot):
    misses = frame[frame.is_miss].copy() if len(frame) else frame.copy()
    rows = []
    if not len(misses):
        for name in ('within_batch','independent_donor','independent_donor_no_recipient_prior'):
            rows.append(dict(condition=name,availability=boot.ratio(misses,[],[]),recipient_misses=0,eligible_events=len(frame)))
        return rows
    flags = dict(within_batch=misses.token_batch.astype(bool),independent_donor=misses.cross_prior.astype(bool),
                 independent_donor_no_recipient_prior=misses.cross_prior.astype(bool)&~misses.token_batch.astype(bool))
    for name,flag in flags.items():
        stat = boot.ratio(misses,flag.astype(float),np.ones(len(misses)))
        difference = boot.ratio(misses,flag.astype(float)-flags['within_batch'].astype(float),np.ones(len(misses)))
        rows.append(dict(condition=name,availability=stat,difference_from_within_batch=difference,
                         recipient_misses=len(misses),eligible_events=len(frame),
                         population='Identical recipient miss events for all three conditions'))
    return rows


def analyze(root, steps=(30,100,200), destination=None, bootstrap_samples=2000,
            permutations=500, seed=2026, allow_partial=False):
    started = time.perf_counter(); root = Path(root).resolve()
    destination = output_directory(root,destination,'reviewer_controls_v1')
    loaded, plan, inputs, availability = read_collection(root,steps,allow_partial)
    configuration = dict(steps=list(steps),bootstrap_samples=bootstrap_samples,permutations=permutations,
                         seed=seed,allow_partial=allow_partial)
    identity = digest(dict(inputs=inputs,configuration=configuration))
    destination.mkdir(parents=True,exist_ok=True)
    previous = destination/'manifest.json'
    if previous.exists() and json.loads(previous.read_text())['identity_sha256'] != identity:
        raise ValueError('Reviewer output belongs to different inputs/configuration; use a new directory')
    summary = {}; tables = {name:[] for name in ('E2','E3','E6_frequency','E6_donor')}
    for step, data in loaded.items():
        frames = data['frames']; ids = sorted(frames)
        frame = pd.concat([f for f in frames.values() if len(f)],ignore_index=True) if any(len(f) for f in frames.values()) else pd.DataFrame()
        boot = ClusterBootstrap(ids,bootstrap_samples,seed+int(step))
        control,pairs,unpaired = paired_controls(frames)
        recipients = [p['recipient'] for p in pairs]
        paired_boot = ClusterBootstrap(recipients,bootstrap_samples,seed+int(step))
        result = dict(E2=e2_controls(control,paired_boot),E3=e3_controls(frame,boot),
                      E6_frequency=e6_frequency(data['raw'],boot,permutations,seed+int(step)),
                      E6_donor=e6_donor(control,paired_boot),pairing=pairs,unpaired_rollouts=unpaired,
                      source_rollout_clusters=len(ids),recipient_rollout_clusters=len(recipients),
                      zero_ordinary_event_rollouts=[r for r in ids if not len(frames[r])],
                      zero_verification_round_rollouts=[r for r in ids if not len(data['rounds'][r])],
                      ordinary_events=len(frame),recipient_events=len(control),checkpoint=data['checkpoint'])
        summary[step] = result
        for name in tables:tables[name].extend(dict(checkpoint_step=int(step),**r) for r in result[name])
    # Reading and analysis must leave every original byte unchanged.
    changed = [p for p,sha in inputs.items() if file_hash(p)!=sha]
    if changed:raise RuntimeError(f'Original collection changed during read-only analysis: {changed}')
    for name,rows in tables.items():
        save_csv(destination/(name+'.csv'),rows); atomic_json(destination/(name+'.json'),rows)
    atomic_json(destination/'summary.json',summary)
    from reviewer_controls.plotting import plot_reviewer
    plot_reviewer(destination,summary)
    report = ['# Offline E2/E3/E6 reviewer controls','',
        'Frozen independent evaluation rollouts, not sequential policy updates. No models were loaded or run.',
        'Ordinary outcomes exclude prefill, EOS, length/depth/beam censoring. History is strictly earlier verification rounds.',
        'E2 uses the recipient half only, with mutually exclusive current-response / donor-only / neither conditions.',
        'All three E2 risks and contrasts use joint common support and identical fixed empirical stratum weights.',
        'Donors are the first sorted half, recipients the second; no donor is reused. An odd last rollout is excluded from paired controls.',
        'Bootstrap units for donor controls are independent donor/recipient pairs, indexed by recipient rollout ID.',
        'E3 preserves exclusive source priority: same response, sibling response, different prompt, none. Primary estimates exclude same response.',
        'E6 null moves only ordinary miss labels within each token and rollout; token-specific totals and all censored positions remain fixed.',
        'Null intervals are permutation quantiles; observed and observed-minus-null intervals resample whole rollouts.',
        'E6 donor percentages all use the identical recipient-miss population. Frequency-null results instead use all rollouts.',
        'Matched bootstrap draws with missing required cells are invalid. Sparse support yields no interval, never imputed results.',
        'These observational controls do not measure a SPARK intervention.','']
    for step,result in summary.items():
        report += [f'## Checkpoint {step}',f'Ordinary events: {result["ordinary_events"]}; source rollouts: {result["source_rollout_clusters"]}; recipient rollouts: {result["recipient_rollout_clusters"]}.']
        for name in ('E2','E3'):
            for row in result[name]:
                label = row.get('distance',row.get('definition','')+' / '+row.get('source',''))
                contrast = 'current_response_minus_independent_only' if name=='E2' else 'exposed_minus_unexposed'
                report.append(f'- {name} {label}: RD {row["contrasts"][contrast]["risk_difference"]}; RR {row["contrasts"][contrast]["relative_risk"]}. {row["conclusion"]} Support: {row["support"]}.')
        for row in result['E6_frequency']:
            report.append(f'- E6 {row["scope"]}: observed {row["observed"]}; null mean {row["null_mean"]}, null 95% {row["null_interval95"]}; observed-minus-null {row["observed_minus_null"]}; permutation p={row["permutation_p_two_sided"]}.')
        for row in result['E6_donor']:report.append(f'- E6 {row["condition"]}: {row["availability"]}; identical recipient miss denominator={row["recipient_misses"]}.')
        report.append('')
    atomic_bytes(destination/'REVIEWER_CONTROLS_REPORT.md',('\n'.join(report)+'\n').encode())
    atomic_json(destination/'manifest.json',dict(identity_sha256=identity,configuration=configuration,
        collection_plan_sha256=digest(plan),inputs=inputs,availability=availability,source_files_preserved=True,
        implementation={str(p.relative_to(Path(__file__).resolve().parents[1])):file_hash(p) for p in Path(__file__).parent.glob('*.py')},
        runtime_s=time.perf_counter()-started,checkpoints=list(summary),zero_model_forwards=True))
    atomic_json(destination/'complete.json',dict(files={p.name:file_hash(p) for p in destination.iterdir() if p.is_file() and p.name!='complete.json'}))
    return dict(output_dir=str(destination),checkpoints=list(summary),runtime_s=time.perf_counter()-started)

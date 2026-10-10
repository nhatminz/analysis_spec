"""Sparse, read-only root-error persistence observations and response-cluster statistics."""
from pathlib import Path
import csv
import io
import os
import random
import numpy as np
import torch
from motivation.state import atomic_json, digest, generation_seed

DISTANCES = tuple(range(1, 6))


def validate_root_alignment(draft, target_token, target_position, cache_length):
    """Draft inputs are shifted by one; both distributions predict AFTER this root."""
    if (int(draft['root_token']) != int(target_token) or
            int(draft['root_position']) != int(target_position) or
            draft['cache_length'] != cache_length):
        raise ValueError('Draft/target root token, logical position or prefix cache length is misaligned')


def match_control(donors, round_index, response_id, seed):
    """Choose only independent donors at EXACTLY the origin round, using private RNG."""
    candidates = [(key, rounds[round_index]) for key, rounds in sorted(donors.items())
                  if key != response_id and round_index < len(rounds) and rounds[round_index] is not None]
    if not candidates:
        return None, None
    rng = random.Random(generation_seed(seed, round_index, response_id))
    return rng.choice(candidates)


class RootObserver:
    """Keep GPU vocab vectors only for the current round; export <=8 IDs and scalars.

    A missing positive-eight set is recorded as None, never padded with zeros.
    Controls are chosen at origin r and evaluated on the recipient context r+d.
    """
    def __init__(self, response_id, donors=None, seed=2026):
        self.response_id = response_id
        self.donors = donors
        self.seed = seed
        self.rounds = []
        self.measurements = []
        self.pending = []

    def observe_round(self, q, p, *, root_token, root_position, cache_length):
        if q.ndim != 1 or p.shape != q.shape:
            raise ValueError('Aligned distributions must share one full vocabulary')
        if q.numel() < 8 or not torch.isfinite(q).all() or not torch.isfinite(p).all():
            raise ValueError('Invalid root probabilities')
        if bool((q < 0).any()) or bool((p < 0).any()):
            raise ValueError('Negative root probabilities')
        # p is exactly the sampler's storage-precision distribution; q is the
        # original untempered proposal softmax. Subtract in FP32 on the device.
        error = p.float() - q.float()
        r = len(self.rounds)
        if self.donors is not None:
            for origin in self.pending:
                d = r-origin['round']
                if d not in DISTANCES:
                    continue
                same = float(error[origin['ids_gpu']].mean()) if origin['ids_gpu'] is not None else None
                control = float(error[origin['control_gpu']].mean()) if origin['control_gpu'] is not None else None
                self.measurements.append(dict(response_id=self.response_id, origin_round=origin['round'],
                    future_round=r, distance=d, same=same, control=control,
                    difference=same-control if same is not None and control is not None else None,
                    donor_response_id=origin['donor_id'],
                    control_status=origin['control_status'],
                    status='paired' if same is not None and control is not None else
                           ('missing_positive_eight' if same is None else origin['control_status'])))
        values, ids = torch.topk(error, 8)
        positive_eight = bool((values > 0).all())
        selected = ids.tolist() if positive_eight else None
        self.rounds.append(dict(round=r, token_ids=selected, root_token=root_token,
                                root_position=root_position, cache_length=cache_length))
        if self.donors is not None:
            donor_id, donor_ids = match_control(self.donors, r, self.response_id, self.seed)
            control_status = ('matched' if donor_ids is not None else
                ('missing_donor_positive_eight' if any(r < len(rounds) for key,rounds in self.donors.items()
                                                     if key != self.response_id) else 'missing_donor_round'))
            self.pending = [x for x in self.pending if r-x['round'] < max(DISTANCES)]
            self.pending.append(dict(round=r, ids_gpu=ids if positive_eight else None,
                control_gpu=torch.tensor(donor_ids, dtype=torch.long, device=q.device) if donor_ids is not None else None,
                donor_id=donor_id,control_status=control_status))

    def finish(self, output, eos_token_id):
        tokens = output['generated_token_ids'][0]
        n = len(self.rounds)
        if output['response_verification_rounds'] != [n]:
            raise ValueError('Observed round count disagrees with acceptance counters')
        if not tokens or len(tokens) != output['response_accepted_length_sum'][0]+1:
            raise ValueError('Emitted tokens and acceptance counters are misaligned')
        if eos_token_id in tokens[:-1]:
            raise ValueError('Generation continued after EOS')
        if n == 0 and tokens != [eos_token_id]:
            raise ValueError('Zero-round response must be prefill EOS')
        if self.donors is not None:
            for origin in self.rounds:
                for d in DISTANCES:
                    if origin['round']+d >= n:
                        self.measurements.append(dict(response_id=self.response_id, origin_round=origin['round'],
                            future_round=origin['round']+d, distance=d, same=None, control=None, difference=None,
                            donor_response_id=None, control_status=None,status='unfinished_horizon'))
        self.pending.clear()
        return dict(response_id=self.response_id, rounds=n, zero_round=n == 0,
                    termination='prefill_eos' if n == 0 else ('eos' if tokens[-1] == eos_token_id else 'token_cap'),
                    generated_tokens=len(tokens), accepted_length_sum=output['response_accepted_length_sum'][0],
                    round_observations=self.rounds, measurements=self.measurements)


def clustered_estimate(rows, response_ids, key, resamples=2000, seed=2026):
    """Round-weighted mean, percentile CI; resample WHOLE evaluation responses."""
    sums = np.zeros(len(response_ids));counts = np.zeros(len(response_ids))
    index = {response_id:i for i,response_id in enumerate(response_ids)}
    for row in rows:
        value = row.get(key)
        if value is not None:
            i = index[row['response_id']];sums[i] += value;counts[i] += 1
    total = int(counts.sum())
    if not total:
        return dict(mean=None, ci95=[None,None], samples=0, responses=0, bootstrap_valid_resamples=0)
    if resamples < 1:
        raise ValueError('Positive bootstrap resample count required')
    rng = np.random.default_rng(seed)
    # Include zero-round/zero-valid responses in the cluster sampling population.
    draws = rng.integers(0, len(response_ids), size=(resamples, len(response_ids)))
    denominator = counts[draws].sum(axis=1)
    numerator = sums[draws].sum(axis=1)
    boot = numerator[denominator > 0]/denominator[denominator > 0]
    ci = np.quantile(boot, [.025,.975]).tolist() if len(boot) else [None,None]
    return dict(mean=float(sums.sum()/total), ci95=ci, samples=total,
                responses=int((counts > 0).sum()), bootstrap_valid_resamples=len(boot))


def summarize(responses, resamples=2000, seed=2026):
    evaluations = [x for x in responses if x['role'] == 'evaluation']
    ids = [x['response_id'] for x in evaluations]
    rows = [r for x in evaluations for r in x['measurements']]
    result = []
    for d in DISTANCES:
        current = [r for r in rows if r['distance'] == d]
        paired = [r for r in current if r['difference'] is not None]
        same = clustered_estimate(current, ids, 'same', resamples, seed)
        result.append(dict(distance=d, same_all=same,
            same=clustered_estimate(paired, ids, 'same', resamples, seed),
            control=clustered_estimate(paired, ids, 'control', resamples, seed),
            paired_difference=clustered_estimate(paired, ids, 'difference', resamples, seed),
            valid_pairs=len(paired), control_coverage=len(paired)/same['samples'] if same['samples'] else None,
            future_contexts=sum(r['status'] != 'unfinished_horizon' for r in current),
            unfinished_horizons=sum(r['status'] == 'unfinished_horizon' for r in current),
            missing_positive_eight=sum(r['status'] == 'missing_positive_eight' for r in current),
            missing_donor_rounds=sum(r['status'] == 'missing_donor_round' for r in current),
            missing_donor_positive_eight=sum(r['status'] == 'missing_donor_positive_eight' for r in current)))
    return dict(distances=result, evaluation_responses=len(evaluations),
        zero_round_evaluation_responses=sum(x['zero_round'] for x in evaluations),
        estimator='round-weighted mean; plotted curves use identical paired support; same_all includes unmatched origins',
        bootstrap='percentile 95% CI, evaluation-response clusters; donors conditioned on their observed sets',
        bootstrap_resamples=resamples, bootstrap_seed=seed,
        interpretation='Descriptive evidence only; inspect paired differences and intervals before assessing persistence.')


def atomic_text(path, text):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def save_results(root, responses, manifest, *, plot=True):
    root = Path(root)
    summary = summarize(responses, manifest['config']['bootstrap_samples'], manifest['seeds']['bootstrap'])
    summary.update(status=manifest['status'], runtime_s=manifest['runtime_s'])
    fields = ['response_id','origin_round','future_round','distance','same','control','difference','donor_response_id','control_status','status']
    buffer = io.StringIO();writer = csv.DictWriter(buffer, fieldnames=fields);writer.writeheader()
    for response in responses:
        writer.writerows(response['measurements'])
    atomic_text(root/'results.csv',buffer.getvalue())
    atomic_json(root/'summary.json',summary)
    manifest['sample_counts'] = dict(donor_responses=sum(x['role']=='donor' for x in responses),
        evaluation_responses=summary['evaluation_responses'], total_rounds=sum(x['rounds'] for x in responses),
        zero_round_responses=sum(x['zero_round'] for x in responses),
        valid_pairs={str(x['distance']):x['valid_pairs'] for x in summary['distances']})
    atomic_json(root/'manifest.json',manifest)
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,4))
        for key,label in [('same','Same response'),('control','Shuffled control')]:
            stats = [x[key] for x in summary['distances']]
            means = np.array([s['mean'] if s['mean'] is not None else np.nan for s in stats])
            lo = [s['ci95'][0] if s['ci95'][0] is not None else np.nan for s in stats]
            hi = [s['ci95'][1] if s['ci95'][1] is not None else np.nan for s in stats]
            line, = ax.plot(DISTANCES,means,marker='o',label=label)
            ax.fill_between(DISTANCES,lo,hi,alpha=.2,color=line.get_color())
        ax.axhline(0,color='grey',linewidth=.7);ax.set_xticks(DISTANCES)
        ax.set_xlabel('Verification-round distance d');ax.set_ylabel('Mean future p(v) − q(v)')
        ax.set_title('Root prediction-error persistence (paired contexts)')
        ax.legend();fig.tight_layout()
        tmp = root/'verification_persistence.pdf.tmp';fig.savefig(tmp,format='pdf');plt.close(fig)
        os.replace(tmp,root/'verification_persistence.pdf')
    return summary


def load_frozen_shadow(model_path, saved, source_config):
    """Reuse FastGRPO architecture and adapter restoration, with ONE pure draft.

    No build_models training optimizers, Reflex projector, or GRPO path is invoked.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model
    from helper.fastgrpo_model import FastGRPOModel
    from motivation.runtime import load_target
    if saved.get('format') != 'simplelr_opd_policy_lag_a1a2_v2' or saved.get('policy_step',0) < 1:
        raise ValueError('An online-trained analysis v2 checkpoint with policy_step >= 1 is required')
    shadow = saved['shadow']['weights']
    if any('opd_' in key for key in shadow):
        raise ValueError('Expected PURE FastGRPO shadow.weights without an OPD projector')
    dtype = {'bf16':torch.bfloat16,'fp16':torch.float16}[source_config.get('dtype','bf16')]
    target = AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,torch_dtype=dtype,
        attn_implementation=source_config.get('attention_implementation','sdpa')).cuda().eval()
    dc = AutoConfig.from_pretrained(model_path,local_files_only=True)
    dc.num_hidden_layers=1;dc.rope_scaling=None;dc.torch_dtype=dtype
    model = FastGRPOModel(dc,target).cuda()
    model.draft_model.load_state_dict(shadow,strict=True)
    target = get_peft_model(target,LoraConfig(task_type=TaskType.CAUSAL_LM,r=64,lora_alpha=32,lora_dropout=0.,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    load_target(target,saved['target_lora']);model.target_model=target
    model.eval();model.requires_grad_(False)
    if digest(model.draft_model.state_dict()) != digest(shadow):
        raise ValueError('Shadow restoration did not preserve checkpoint weights')
    return model

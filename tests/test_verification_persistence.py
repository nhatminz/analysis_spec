import json
import numpy as np
import pytest
import torch
from motivation.verification_persistence import (RootObserver, validate_root_alignment,
    match_control, clustered_estimate, summarize, save_results)


def distribution(start=0):
    q = torch.full((32,),1/32)
    p = torch.full((32,),.36/24);p[start:start+8] = .08
    return q,p


def observe(observer,start=0):
    q,p = distribution(start)
    observer.observe_round(q,p,root_token=9,root_position=len(observer.rounds)+3,cache_length=len(observer.rounds)+3)


def finish(observer,eos=False):
    n = len(observer.rounds)
    return observer.finish(dict(generated_token_ids=[[96] if n==0 else [1]*n+[96 if eos else 2]],
        response_verification_rounds=[n],response_accepted_length_sum=[n]),96)


def test_alignment_checks_token_position_and_prefix_cache():
    draft = dict(root_token=torch.tensor(9),root_position=torch.tensor(12),cache_length=12)
    validate_root_alignment(draft,9,12,12)
    for args in [(8,12,12),(9,11,12),(9,12,11)]:
        with pytest.raises(ValueError,match='misaligned'):validate_root_alignment(draft,*args)


def test_control_is_independent_exact_round_and_deterministic():
    donors = {'donor_a':[list(range(8)),None], 'donor_b':[None,list(range(8,16))],
              'eval':[list(range(24,32))]*4}
    assert match_control(donors,0,'eval',2026) == ('donor_a',list(range(8)))
    assert match_control(donors,1,'eval',2026) == ('donor_b',list(range(8,16)))
    assert match_control(donors,2,'eval',2026) == (None,None)
    assert match_control(donors,0,'eval',2026) == match_control(donors,0,'eval',2026)


def test_future_indexing_control_uses_recipient_context_and_missing_horizons():
    donor_sets = {'donor':[list(range(8,16))]*7}
    observer = RootObserver('eval',donor_sets)
    observe(observer,0)
    for _ in range(6):observe(observer,8)
    response = finish(observer)
    rows = [r for r in response['measurements'] if r['origin_round']==0]
    assert sorted(r['distance'] for r in rows) == [1,2,3,4,5]
    assert all(r['future_round']==r['distance'] for r in rows)
    q,p = distribution(8)
    assert all(r['same']==pytest.approx(float((p-q)[:8].mean())) for r in rows)
    assert all(r['control']==pytest.approx(float((p-q)[8:16].mean())) for r in rows)
    assert all(r['difference']<0 for r in rows)
    last = [r for r in response['measurements'] if r['origin_round']==6]
    assert len(last)==5 and all(r['status']=='unfinished_horizon' and r['same'] is None for r in last)
    assert len(response['measurements'])==7*5
    assert not observer.pending


def test_missing_donor_round_never_fabricated():
    observer = RootObserver('eval',{'donor':[]})
    observe(observer);observe(observer)
    response = finish(observer)
    row = next(r for r in response['measurements'] if r['future_round']==1)
    assert row['same']>0 and row['control'] is None and row['difference'] is None
    assert row['status']=='missing_donor_round'
    response['role']='evaluation'
    summary = summarize([response],100)
    assert summary['distances'][0]['same_all']['samples']==1
    assert summary['distances'][0]['valid_pairs']==0
    assert summary['distances'][0]['control_coverage']==0


def test_fewer_than_eight_positive_tokens_is_missing_not_zero_padded():
    observer = RootObserver('eval',{})
    q = torch.full((32,),1/32)
    observer.observe_round(q,q,root_token=0,root_position=0,cache_length=0)
    observe(observer)
    assert observer.rounds[0]['token_ids'] is None
    assert observer.measurements[0]['same'] is None
    assert observer.measurements[0]['status']=='missing_positive_eight'


def test_unusable_donor_set_is_distinct_from_missing_donor_round():
    observer = RootObserver('eval',{'donor':[None]})
    observe(observer);observe(observer)
    assert observer.measurements[0]['status']=='missing_donor_positive_eight'
    assert observer.measurements[0]['control'] is None


def test_bootstrap_matches_whole_cluster_reference_with_unequal_cluster_sizes():
    rows = [dict(response_id='a',same=1.),dict(response_id='a',same=3.),dict(response_id='b',same=10.)]
    result = clustered_estimate(rows,['a','b','empty'],'same',2000,2026)
    draws = np.random.default_rng(2026).integers(0,3,size=(2000,3))
    sums = np.array([4.,10.,0.])[draws].sum(1)
    counts = np.array([2.,1.,0.])[draws].sum(1)
    expected = sums[counts>0]/counts[counts>0]
    assert result['mean']==pytest.approx(14/3)
    assert result['ci95']==pytest.approx(np.quantile(expected,[.025,.975]))
    assert result['samples']==3 and result['responses']==2
    assert result['bootstrap_valid_resamples']==len(expected)
    assert clustered_estimate([],[],'same')['mean'] is None


def test_eos_zero_round_and_verified_eos():
    empty = RootObserver('empty',{})
    result = finish(empty,eos=True)
    assert result['termination']=='prefill_eos' and result['zero_round']
    assert result['measurements']==[]
    active = RootObserver('active',{});observe(active)
    result = finish(active,eos=True)
    assert result['termination']=='eos' and not result['zero_round']
    assert all(r['status']=='unfinished_horizon' for r in result['measurements'])
    with pytest.raises(ValueError,match='continued after EOS'):
        active.finish(dict(generated_token_ids=[[96,2]],response_verification_rounds=[1],response_accepted_length_sum=[1]),96)
    with pytest.raises(ValueError,match='round count'):
        active.finish(dict(generated_token_ids=[[1,2]],response_verification_rounds=[2],response_accepted_length_sum=[1]),96)


def test_output_empty_partial_and_paired_support(tmp_path):
    manifest = dict(config=dict(bootstrap_samples=2000),seeds=dict(bootstrap=2026),status='wall_clock_budget_exceeded',runtime_s=1.)
    result = save_results(tmp_path,[],manifest)
    assert result['evaluation_responses']==0
    assert (tmp_path/'verification_persistence.pdf').read_bytes().startswith(b'%PDF')
    assert json.loads((tmp_path/'manifest.json').read_text())['status']=='wall_clock_budget_exceeded'
    assert (tmp_path/'results.csv').read_text().startswith('response_id,')


def test_sampler_observation_uses_exact_filtered_distribution_without_rng_change():
    from helper.fastgrpo_generate import sampling
    logits = torch.tensor([[[2.,1.,.2,-3.]]])
    torch.manual_seed(91);plain = sampling(logits,top_p=.9,temperature=.7);plain_state=torch.get_rng_state()
    seen=[]
    torch.manual_seed(91)
    observed = sampling(logits,top_p=.9,temperature=.7,probability_observer=lambda p,i:seen.append(p.clone()))
    assert torch.equal(plain,observed) and torch.equal(plain_state,torch.get_rng_state())
    probs = (logits[0,0]/.7).softmax(-1)
    sorted_p,indices = probs.sort(descending=True)
    mask = (sorted_p.cumsum(-1)>.9).roll(1);mask[0]=False
    sorted_p[mask]=0;sorted_p/=sorted_p.sum()
    expected=torch.zeros_like(probs).scatter(0,indices,sorted_p)
    assert torch.equal(seen[0][0],expected)

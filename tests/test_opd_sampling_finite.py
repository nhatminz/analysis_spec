"""Parity and host-read checks for the OPD-only finite sampler."""
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from helper import opd_sampling

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'tests/references/opd_sampling_before_finite.py'
spec = importlib.util.spec_from_file_location('opd_sampler_before_finite', REFERENCE)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA sampler parity')


def capture(tokens, probs, metadata):
    return tuple(x.clone() for x in metadata) if metadata is not None else None


def assert_outputs_equal(a, b):
    assert torch.equal(a[0], b[0])
    if a[1] is None:
        assert b[1] is None
    else:
        assert torch.equal(a[1], b[1])
    if a[2] is None:
        assert b[2] is None
    else:
        assert len(a[2]) == len(b[2])
        for x, y in zip(a[2], b[2]):
            assert torch.equal(x, y)


@CUDA
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('shape', [(1, 1, 97), (2, 7, 257), (3, 1, 31), (2, 2, 151936)])
@pytest.mark.parametrize('top_p,top_k', [(None, None), (0., 0), (.95, None), (1., None),
                                       (None, 1), (None, 16), (.9, 8), (.1, 16)])
def test_finite_tokens_probabilities_sort_metadata_rng_bitwise(dtype, shape, top_p, top_k):
    gen = torch.Generator(device='cuda').manual_seed(51)
    logits = torch.randn(shape, device='cuda', dtype=dtype, generator=gen)
    for temperature in (.4, 1., 1.7):
        for seed in (0, 42, 1234567):
            kwargs = dict(top_p=top_p, top_k=top_k, temperature=temperature,
                          eos_token_id=2, metadata_builder=capture)
            torch.manual_seed(seed)
            before = old.sampling(logits, **kwargs)
            expected_rng = torch.cuda.get_rng_state()
            expected_cpu_rng = torch.get_rng_state()
            torch.manual_seed(seed)
            after = opd_sampling.sampling(logits, **kwargs, mode='finite')
            assert_outputs_equal(before, after)
            assert torch.equal(expected_rng, torch.cuda.get_rng_state())
            assert torch.equal(expected_cpu_rng, torch.get_rng_state())


@CUDA
@pytest.mark.parametrize('kind', ['ties', 'strided'])
def test_finite_tied_and_noncontiguous_logits(kind):
    logits = (torch.zeros(2, 3, 97, device='cuda', dtype=torch.bfloat16) if kind == 'ties'
              else torch.randn(2, 3, 194, device='cuda')[:, :, ::2])
    for seed in (1, 42, 987):
        torch.manual_seed(seed)
        before = old.sampling(logits, top_p=.95, top_k=16, metadata_builder=capture)
        rng = torch.cuda.get_rng_state()
        torch.manual_seed(seed)
        after = opd_sampling.sampling(logits, top_p=.95, top_k=16,
                                      metadata_builder=capture, mode='finite')
        assert_outputs_equal(before, after)
        assert torch.equal(rng, torch.cuda.get_rng_state())


@CUDA
@pytest.mark.parametrize('top_p,top_k', [(None, None), (.95, None), (None, 16), (.9, 8)])
@pytest.mark.parametrize('kind', ['nan', 'posinf', 'neginf', 'all_nan', 'all_inf', 'mixed'])
def test_strict_nan_inf_all_invalid_and_mixed_fallback_matches_old(kind, top_p, top_k):
    logits = torch.randn(2, 3, 97, device='cuda')
    if kind == 'nan': logits[0, 0, 3] = float('nan')
    elif kind == 'posinf': logits[0, 0, 3] = float('inf')
    elif kind == 'neginf': logits[0, 0, 3] = -float('inf')
    elif kind == 'all_nan': logits.fill_(float('nan'))
    elif kind == 'all_inf': logits.fill_(float('inf'))
    else:
        logits[0, 0] = float('nan')
        logits[0, 1] = -float('inf')
        logits[1, 0, 5] = float('inf')
    kwargs = dict(top_p=top_p, top_k=top_k, metadata_builder=capture)
    torch.manual_seed(67)
    before = old.sampling(logits, **kwargs)
    rng = torch.cuda.get_rng_state()
    torch.manual_seed(67)
    after = opd_sampling.sampling(logits, **kwargs, mode='strict')
    assert_outputs_equal(before, after)
    assert torch.equal(rng, torch.cuda.get_rng_state())


@CUDA
@pytest.mark.parametrize('top_p,top_k', [(None, None), (.95, None), (None, 8), (.9, 8)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_actual_teacher_metadata_and_feedback_parity(dtype, top_p, top_k):
    from helper.opd_reflex import OPDReflex
    from helper.tree_verification import PackedTree
    from test_opd_reflex import seed as seed_adapter
    gen = torch.Generator(device='cuda').manual_seed(12)
    head = torch.nn.Linear(32, 97, bias=False, device='cuda', dtype=dtype)
    model = SimpleNamespace(lm_head=head, opd_projector=torch.randn(32, 8, device='cuda', generator=gen) * .01)
    identity = torch.arange(97, device='cuda')
    states = [OPDReflex(train_projector=True) for _ in range(2)]
    adapter_rows = torch.randn(16, 8, device='cuda', generator=gen) * .02
    hidden = torch.randn(2, 1, 32, device='cuda', dtype=dtype, generator=gen)
    logits = torch.randn(2, 1, 97, device='cuda', dtype=dtype, generator=gen)
    root = torch.full((2, 1), -1, device='cuda', dtype=torch.long)
    tree = PackedTree(root, root, torch.zeros_like(root), 0)
    path = SimpleNamespace(packed_indices=torch.zeros_like(root))
    answers = []
    updates = []
    for i, state in enumerate(states):
        state.start(model, 2, identity, 32, max_contexts=1, max_nodes=2, max_path=1, max_proposal_contexts=1)
        seed_adapter(state, torch.arange(16, device='cuda'), adapter_rows)
        state.propose(head(hidden), hidden, 8, identity)
        model.opd_projector_grad_sum.zero_(); model.opd_projector_grad_weight.zero_()
        def teacher(tokens, probs, metadata):
            return state.prepare_compact_teacher(tree, path, probs, metadata)
        torch.manual_seed(981)
        sampler = old.sampling if i == 0 else opd_sampling.sampling
        kwargs = {} if i == 0 else dict(mode='finite')
        answer = sampler(logits, top_p=top_p, top_k=top_k, metadata_builder=teacher,
                         return_probs=False, **kwargs)
        answers.append((answer, torch.cuda.get_rng_state()))
        state.feedback(tree, path, None, sampling_metadata=answer[2])
        updates.append((state.B_fast.clone(), model.opd_projector_grad_sum.clone(),
                        model.opd_projector_grad_weight.clone(), state.counters.clone()))
    assert_outputs_equal(answers[0][0], answers[1][0])
    assert torch.equal(answers[0][1], answers[1][1])
    for x, y in zip(updates[0], updates[1]):
        # Identical teacher inputs; atomic addition may reorder across launches.
        torch.testing.assert_close(x, y, rtol=3e-6, atol=1e-8)


@CUDA
@pytest.mark.parametrize('top_p,top_k', [(None, None), (.95, None), (None, 16), (.9, 16)])
def test_finite_sampler_has_no_host_tensor_reads_or_dynamic_compaction(top_p, top_k):
    logits = torch.randn(2, 3, 97, device='cuda')
    for _ in range(3):
        opd_sampling.sampling(logits, top_p=top_p, top_k=top_k, mode='finite')
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        opd_sampling.sampling(logits, top_p=top_p, top_k=top_k, mode='finite')
    names = {event.key for event in profile.key_averages()}
    forbidden = {'aten::item', 'aten::_local_scalar_dense', 'aten::nonzero', 'aten::is_nonzero'}
    assert not (names & forbidden), names & forbidden
    assert 'aten::_assert_async' in names
    assert names.isdisjoint({'cudaDeviceSynchronize', 'cudaStreamSynchronize'})


@CUDA
@pytest.mark.parametrize('top_p,top_k', [(None, None), (.95, None), (None, 16), (.9, 16)])
def test_finite_sampler_is_cuda_graph_capturable(top_p, top_k):
    # Capture rejects scalar host reads/nonzero/device synchronization. This is
    # a validation only: runtime execution continues to use normal eager calls.
    logits = torch.randn(2, 3, 97, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            opd_sampling.sampling(logits, top_p=top_p, top_k=top_k, mode='finite')
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = opd_sampling.sampling(logits, top_p=top_p, top_k=top_k, mode='finite')
    graph.replay()
    assert output[0].shape == (2, 3)


def test_default_keeps_strict_and_strict_body_is_unchanged():
    assert opd_sampling.SAMPLER_MODE == 'strict'
    new_tree = ast.parse(Path(opd_sampling.__file__).read_text())
    old_tree = ast.parse(REFERENCE.read_text())
    new = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sampling_strict')
    reference = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sampling')
    new.name = reference.name
    assert ast.dump(new, include_attributes=False) == ast.dump(reference, include_attributes=False)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_finite_guard_fails_instead_of_silently_sampling_invalid_logits_cpu(value):
    # CUDA assertion poisons its context; CPU tests exercise the same contract.
    with pytest.raises(RuntimeError, match='finite logits'):
        opd_sampling.sampling(torch.full((1, 1, 97), value), mode='finite')


@CUDA
@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_cuda_finite_guard_rejects_nonfinite_input_in_isolated_process(value):
    # A device assertion invalidates the CUDA context, so isolate each case.
    code = '''import sys, torch
from helper.opd_sampling import sampling
x=torch.zeros(2,3,97,device='cuda')
x[0,0,7]=float(sys.argv[1])
sampling(x,mode='finite')
torch.cuda.synchronize()
'''
    result = subprocess.run([sys.executable, '-c', code, value], cwd=ROOT,
                            env=os.environ.copy(), text=True, capture_output=True, timeout=60)
    assert result.returncode != 0
    assert 'device-side assert' in result.stderr or 'finite logits' in result.stderr


@CUDA
@pytest.mark.parametrize('stream', [False, True])
def test_full_opd_rollout_tokens_rng_acceptance_and_forward_parity(monkeypatch, stream):
    from test_fastgrpo_rewrite import tiny, run
    from helper.specualtive_generate import speculative_generate
    answers = []
    for mode in ('strict', 'finite'):
        monkeypatch.setattr(opd_sampling, 'SAMPLER_MODE', mode)
        model = tiny(); model.enable_opd(8)
        answers.append(run(speculative_generate, model, method='opd_reflex', opd_update_stream=stream))
    a, b = answers
    for key in ('generated_token_ids', 'total_acc_length', 'total_decoded_token_num',
                'verification_batches', 'total_accepted_draft_tokens'):
        assert a[0][key] == b[0][key]
    assert torch.equal(a[1], b[1])
    assert a[2] == b[2]
    for x, y in zip(a[3], b[3]): assert torch.equal(x, y)

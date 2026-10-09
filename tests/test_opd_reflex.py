import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from helper.opd_reflex import OPDReflex, initialize_projector, select_states_reference, union_reference
from helper.tree_verification import PackedTree, VerifiedPath
DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])

def digest(xs):
    h = hashlib.sha256()
    for x in xs:
        h.update(str(tuple(x.shape)).encode())
        h.update(x.detach().cpu().float().numpy().tobytes())
    return h.hexdigest()

def state(device='cpu', v=257, h=32, enabled=True, lr=0.01, topk=16):
    generator = torch.Generator().manual_seed(31)
    head = torch.nn.Linear(h, v, bias=False).to(device)
    head.weight.data.copy_(torch.randn(v, h, generator=generator).to(device) * 0.1)
    model = SimpleNamespace(lm_head=head, opd_projector=initialize_projector(h, 8).to(device))
    s = OPDReflex(8, topk, lr, enabled=enabled)
    mapping = torch.arange(v, device=device)
    s.start(model, 3, mapping, h, max_contexts=8, max_nodes=24, max_path=5, max_proposal_contexts=4)
    return (s, model, mapping)

def seed(s, tokens, weights):
    s.host_active_count = len(tokens)
    s.B_fast.zero_()
    s.B_fast[tokens] = weights
    s.active_count.fill_(len(tokens))
    s.active_ids[:len(tokens)] = tokens
    s.bitmap.zero_()
    for token in tokens.cpu().tolist():
        s.bitmap[token // 32] |= torch.tensor(1 << token % 32, device=s.bitmap.device, dtype=torch.int64).to(torch.int32)
    s._ever_updated = bool(len(tokens))

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('slots', [0, 1, 8, 32])
@pytest.mark.parametrize('ties', [False, True])
def test_sparse_proposal_exact_ids_and_dense_probability_oracle(device, slots, ties):
    (s, model, mapping) = state(device)
    tokens = torch.arange(slots, device=device) * 7
    seed(s, tokens, torch.randn(slots, 8, device=device) * 0.01)
    hidden = torch.randn(3, 4, 32, device=device)
    raw = torch.zeros(3, 4, 257, device=device) if ties else torch.randn(3, 4, 257, device=device)
    (q, ids, _) = s.propose(raw, hidden, 8, mapping, context_offset=0)
    u = s.u_cache[:3, :4]
    dense = raw.float() + u.matmul(s.B_fast.t())
    expected = torch.argsort(dense, dim=-1, descending=True, stable=True)[..., :8]
    assert torch.equal(ids, expected)
    torch.testing.assert_close(q, dense.softmax(-1).gather(-1, expected), rtol=2e-05, atol=2e-07)
    if slots == 0:
        (off, _, _) = state(device, enabled=False)
        a = off.propose(raw, hidden, 8, mapping)
        assert torch.equal(q, a[0]) and torch.equal(ids, a[1])

@pytest.mark.parametrize('device', DEVICES)
def test_state_selection_only_visited_and_expanded_one_hop_frontier(device):
    parents = torch.tensor([[-1, 0, 0, 0, 1, 1, 2, 5]], device=device)
    contexts = torch.tensor([[0, 1, 2, -1, 3, 4, 5, -1]], device=device)
    tree = PackedTree(parents, parents.clone(), contexts, 3)
    path = SimpleNamespace(packed_indices=torch.tensor([[0, 1, 5, 7, -1]], device=device))
    (expected_weights, expected_kind) = select_states_reference(tree, path)
    assert expected_kind.tolist() == [[1, 1, 2, 0, 2, 1, 0, 0]]
    if device != 'cpu':
        from helper.opd_reflex_kernels import select_states
        w = torch.empty_like(expected_weights)
        kind = torch.empty_like(expected_kind, dtype=torch.int32)
        select_states(tree, path, w, kind, 1.0, 1.0)
        assert torch.equal(w, expected_weights) and torch.equal(kind, expected_kind)

def test_union_has_unique_members_and_tail_without_shortlist_renormalization():
    p = torch.tensor([[0.1, 0.2, 0.3, 0.15, 0.25]])
    q = torch.tensor([[0.2, 0.1, 0.25, 0.1, 0.35]])
    (ids, valid, pu, qu, pt, qt, kl) = union_reference(p, q, torch.tensor([[2, 4]]), torch.tensor([[0, 4]]))
    assert ids[valid].tolist() == [0, 4, 2]
    assert pt.item() == pytest.approx(0.35) and qt.item() == pytest.approx(0.2)
    expected = (torch.tensor([0.1, 0.25, 0.3, 0.35]) * (torch.tensor([0.1, 0.25, 0.3, 0.35]) / torch.tensor([0.2, 0.35, 0.25, 0.2])).log()).sum()
    torch.testing.assert_close(kl[0], expected)

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('invalid', [False, True])
def test_shared_weighted_update_union_tail_selected_head_and_counters_match_dense_oracle(device, invalid):
    from helper.opd_reflex import OPD_COUNTER_NAMES
    (s, model, mapping) = state(device, v=13, h=32, lr=0.05, topk=4)
    s.visited_weight = 2.0
    s.frontier_weight = 0.5
    hidden = torch.randn(3, 3, 32, device=device)
    raw = model.lm_head(hidden)
    s.propose(raw, hidden, 3, mapping, context_offset=0)
    parents = torch.tensor([[-1, 0, 0]] * 3, device=device)
    tree = PackedTree(parents, parents.clone(), torch.tensor([[0, 1, 2]] * 3, device=device), 1)
    path = SimpleNamespace(packed_indices=torch.tensor([[0, 1, -1]] * 3, device=device))
    target = torch.randn(3, 3, 13, device=device).softmax(-1)
    if invalid:
        target[0, 0].zero_()
    (weights, kind) = select_states_reference(tree, path, 2.0, 0.5)
    p = target[..., mapping].float()
    mass = p.sum(-1)
    good = (mass > 0) & torch.isfinite(mass)
    p = p / torch.where(good, mass, 1.0)[..., None]
    weights = torch.where(good, weights, 0.0).reshape(-1)
    norm = s.norm_cache[:3, :3].clone()
    u = s.u_cache[:3, :3].clone()
    q = (raw.float() - norm[..., 0, None]).exp() / norm[..., 1, None]
    ti = torch.argsort(p, descending=True, stable=True)[..., :4].reshape(-1, 4)
    di = s.ids_cache[:3, :3].clone().reshape(-1, 4)
    (union, valid, pu, qu, pt, qt, kl) = union_reference(p.reshape(-1, 13), q.reshape(-1, 13), ti, di)
    qu[:, :4] = s.q_cache[:3, :3].clone().reshape(-1, 4)
    gradients = torch.where(valid, (qu - pu) * weights[:, None], 0.0)
    expected = torch.zeros_like(s.B_fast)
    expected.index_add_(0, union.flatten(), (gradients[..., None] * u.reshape(-1, 8)[:, None, :]).reshape(-1, 8))
    expected *= -s.fast_lr / weights.sum()
    before_u = s.u_cache.clone()
    s.feedback(tree, path, target)
    torch.testing.assert_close(s.B_fast, expected, rtol=4e-05, atol=2e-07)
    assert torch.equal(s.u_cache.view(torch.int32), before_u.view(torch.int32))
    counters = dict(zip(OPD_COUNTER_NAMES, s.counters.cpu().tolist()))
    assert counters['opd_selected_states'] == 9 - int(invalid)
    assert counters['opd_visited_states'] == 6 - int(invalid) and counters['opd_frontier_states'] == 3
    assert counters['opd_invalid_states'] == int(invalid) and counters['opd_updates'] == 1
    assert counters['opd_state_weight'] == weights.sum().item()
    torch.testing.assert_close(torch.tensor(counters['opd_compact_mass_sum']), mass[good].cpu().sum(), rtol=1e-06, atol=1e-06)
    assert counters['opd_kl_sum'] == pytest.approx(torch.where(weights > 0, weights * kl, 0.0).sum().item(), rel=3e-05, abs=2e-06)
    s.clear()
    assert not s.B_fast.any()

@pytest.mark.parametrize('device', DEVICES)
def test_feedback_improves_future_fixed_state_without_any_extra_transformer_forward(device):
    (s, model, mapping) = state(device, v=5, h=32, lr=0.1, topk=4)
    model.lm_head.weight.data.zero_()
    model.opd_projector.zero_()
    model.opd_projector[0, 0] = 1.0
    h = torch.zeros(3, 1, 32, device=device)
    h[..., 0] = 1.0
    raw = torch.zeros(3, 1, 5, device=device)
    (before, ids, _) = s.propose(raw, h, 2, mapping, root=True)
    target_token = min(set(range(5)) - set(ids[0, 0].cpu().tolist()))
    assert not (ids == target_token).any()
    tree = PackedTree(torch.full((3, 1), -1, device=device, dtype=torch.long), torch.full((3, 1), -1, device=device, dtype=torch.long), torch.zeros(3, 1, device=device, dtype=torch.long), 0)
    path = SimpleNamespace(packed_indices=torch.zeros(3, 1, device=device, dtype=torch.long))
    teacher = torch.zeros(3, 1, 5, device=device)
    teacher[..., target_token] = 1.0
    s.feedback(tree, path, teacher)
    (after, ids, _) = s.propose(raw, h, 2, mapping, root=True)
    assert (ids == target_token).any(-1).all()
    q4 = (after * (ids == target_token)).sum(-1)
    assert (q4 > 0.2).all()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA retained zero-row identity')
def test_zero_rows_are_retained_without_changing_proposals():
    (s, _, mapping) = state('cuda:0', v=521)
    tokens = torch.arange(256, device='cuda')
    seed(s, tokens, torch.randn(256, 8, device='cuda'))
    s.B_fast.zero_()
    s.round_weight.fill_(1.0)
    from helper.opd_reflex_kernels import _round_end
    _round_end[1,](s.B_fast, s.bitmap, s.active_ids, s.active_count, s.round_weight, s.counters, 8, 0.01, 128, 8, num_warps=4)
    assert s.bitmap.any() and s.active_count.item() == 256
    raw = torch.randn(3, 4, 521, device='cuda')
    h = torch.randn(3, 4, 32, device='cuda')
    (off, _, _) = state('cuda:0', v=521, enabled=False)
    for (a, b) in zip(s.propose(raw, h, 8, mapping), off.propose(raw, h, 8, mapping)):
        assert torch.equal(a, b)

@pytest.mark.skipif(not torch.cuda.is_available(), reason='native BF16 selected-row head oracle')
@pytest.mark.parametrize('hidden_size', [32, 2048])
def test_gpu_selected_head_native_bf16_matches_dense_rows(hidden_size):
    (s, model, mapping) = state('cuda:0', v=521, h=hidden_size, lr=0.0)
    model.lm_head.bfloat16()
    model.lm_head.weight.data.mul_(0.1)
    s._layout = None
    s.start(model, 3, mapping, hidden_size, max_contexts=8, max_nodes=24, max_path=5, max_proposal_contexts=4)
    h = torch.randn(3, 3, hidden_size, device='cuda', dtype=torch.bfloat16)
    raw = model.lm_head(h)
    s.propose(raw, h, 8, mapping)
    parents = torch.tensor([[-1, 0, 0]] * 3, device='cuda')
    tree = PackedTree(parents, parents.clone(), torch.tensor([[0, 1, 2]] * 3, device='cuda'), 1)
    path = SimpleNamespace(packed_indices=torch.tensor([[0, 1, -1]] * 3, device='cuda'))
    target = torch.randn(3, 3, 521, device='cuda').softmax(-1)
    norm = s.norm_cache[:3, :3].clone()
    s.feedback(tree, path, target)
    teacher_ids = s.teacher_ids[:9 * 16].view(3, 3, 16)
    expected = (raw.float().gather(-1, teacher_ids) - norm[..., 0, None]).exp() / norm[..., 1, None]
    actual = s.teacher_q[:9 * 16].view(3, 3, 16)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=1e-06)

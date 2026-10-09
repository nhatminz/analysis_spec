import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from helper.opd_reflex import OPDReflex, initialize_projector, union_reference
from helper.opd_scheduling import schedule, compact_suffix_inplace
from helper.tree_verification import PackedTree, VerifiedPath
from helper.fastgrpo_model import FastGRPOModel
from test_opd_reflex import state, seed, digest
DEVICES = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('nonzero', [1, 3, 8])
def test_target_top16_zero_probability_candidates_never_added(device, nonzero):
    (s, model, mapping) = state(device, v=37, h=32, lr=0.01, topk=16)
    h = torch.randn(3, 1, 32, device=device)
    raw = model.lm_head(h)
    s.propose(raw, h, 8, mapping, root=True)
    root = torch.full((3, 1), -1, device=device, dtype=torch.long)
    tree = PackedTree(root, root.clone(), torch.zeros_like(root), 0)
    path = SimpleNamespace(packed_indices=torch.zeros_like(root))
    p = torch.zeros(3, 1, 37, device=device)
    p[..., mapping[-nonzero:]] = 1.0 / nonzero
    s.feedback(tree, path, p)
    if device != 'cpu':
        ids = s.teacher_ids[:3 * 16].view(3, 16)
        assert (ids >= 0).sum(-1).tolist() == [nonzero] * 3
        assert (s.union_ids[:3 * 32].view(3, 32)[:, 16:] >= 0).sum(-1).max() <= nonzero
    tp = p[..., mapping].reshape(3, 37)
    ti = torch.argsort(tp, descending=True, stable=True)[..., :16]
    di = s.ids_cache[:3, :1].reshape(3, 16)
    (_, valid, _, _, pt, qt, kl) = union_reference(tp, raw.float().softmax(-1).reshape(3, 37), ti, di)
    assert valid[:, 16:].sum(-1).max() <= nonzero
    assert torch.isfinite(kl).all()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='exact adaptive CUDA GEMM/sparse parity')
@pytest.mark.parametrize('slots', [0, 1, 16, 128, 257])
@pytest.mark.parametrize('ties', [False, True])
def test_sparse_dense_switch_bitwise_logits_probabilities_ids(slots, ties):
    (s, _, mapping) = state('cuda:0', v=257)
    tokens = torch.arange(slots, device='cuda')
    weights = torch.randn(slots, 8, device='cuda') * 0.1
    seed(s, tokens, weights)
    raw = torch.zeros(3, 4, 257, device='cuda') if ties else torch.randn(3, 4, 257, device='cuda')
    h = torch.randn(3, 4, 32, device='cuda')
    results = []
    scores = []
    for mode in ('sparse', 'dense', 'sparse'):
        s.proposal_mode = mode
        (q, ids, _) = s.propose(raw, h, 8, mapping)
        results.append((q.clone(), ids.clone()))
        if slots and mode == 'sparse':
            scores.append(s.sparse_scores[:3 * 4 * s.sparse_capacity].view(3, 4, s.sparse_capacity)[..., :slots].clone())
    for result in results[1:]:
        for (a, b) in zip(results[0], result):
            assert torch.equal(a, b)
    if slots:
        assert all((torch.equal(scores[0], x) for x in scores[1:]))
        u = s.u_cache[:3, :4]
        canonical = torch.zeros(3, 4, 257, device='cuda')
        for r in range(8):
            canonical += u[..., r, None] * s.B_fast[:, r]
        z = raw.float() + canonical
        torch.testing.assert_close(results[0][0], z.softmax(-1).gather(-1, results[0][1]), rtol=2e-05, atol=2e-07)
    else:
        expected = raw.float().softmax(-1).gather(-1, results[0][1])
        torch.testing.assert_close(results[0][0], expected, rtol=2e-05, atol=2e-07)

@pytest.mark.parametrize('device', DEVICES)
def test_scheduling_packet_and_inplace_suffix_exact_reference(device):
    tokens = torch.tensor([[4, 5, 6, -1], [7, -1, -1, -1], [8, 9, -1, -1]], device=device)
    indices = torch.tensor([[0, 1, 4, -1], [0, -1, -1, -1], [0, 3, -1, -1]], device=device)
    lengths = torch.tensor([3, 1, 2], device=device)
    path = VerifiedPath(tokens, indices, indices.clone(), lengths)
    workspace = [torch.empty(3, 4, device=device, dtype=torch.bool if i == 2 else torch.long) for i in range(3)] + [torch.empty(3, 1, device=device, dtype=torch.long)]
    packet = torch.empty(3, 7, device=device, dtype=torch.long)
    kernels = None
    if device != 'cpu':
        from helper import tree_kernels as kernels
    (rows, t, i, m, last, width) = schedule(path, 5, 16, workspace, packet, kernels)
    expected = path.padded_gpu(5, 3, 16)
    for (a, b) in zip((t, i, m, last), expected):
        assert torch.equal(a, b)
    assert width == 3 and [r[0] for r in rows] == [3, 1, 2]
    key = torch.randn(3, 2, 12, 8, device=device)
    value = torch.randn_like(key)
    (old_key, old_value) = (key.clone(), value.clone())
    model = SimpleNamespace()
    extension = min((r[2] for r in rows))
    (a, b) = compact_suffix_inplace(key, value, i, 5, extension, model)
    for (actual, old) in ((a, old_key), (b, old_value)):
        ref = torch.cat((old[..., :5, :], old.gather(2, i[:, None, :, None].expand(3, 2, 3, 8))), 2)
        assert torch.equal(actual, ref)
        assert actual.untyped_storage().data_ptr() == key.untyped_storage().data_ptr() if actual is a else actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()

@pytest.mark.parametrize('device', DEVICES)
def test_projector_gradient_accumulates_no_round_parameter_update_then_boundary(device):
    (s, model, mapping) = state(device, v=37, lr=0.02, topk=4)
    s.train_projector = True
    s._layout = None
    s.start(model, 3, mapping, 32, max_contexts=8, max_nodes=24, max_path=5, max_proposal_contexts=4)
    old_a = model.opd_projector.clone()
    h = torch.randn(3, 1, 32, device=device)
    raw = model.lm_head(h)
    root = torch.full((3, 1), -1, device=device, dtype=torch.long)
    tree = PackedTree(root, root, torch.zeros_like(root), 0)
    path = SimpleNamespace(packed_indices=torch.zeros_like(root))
    teacher = torch.randn(3, 1, 37, device=device).softmax(-1)
    s.propose(raw, h, 3, mapping, root=True)
    s.feedback(tree, path, teacher)
    s.propose(raw, h, 3, mapping, root=True)
    old_b = s.B_fast.clone()
    s.feedback(tree, path, teacher)
    assert torch.equal(model.opd_projector, old_a)
    if device != 'cpu':
        ids = s.union_ids[:3 * 8].view(3, 8)
        g = s.union_g[:3 * 8].view(3, 8)
        v = (g[..., None] * old_b[ids.clamp_min(0)] * (ids >= 0)[..., None]).sum(1)
        expected = h[:, 0].float().t().matmul(v)
        torch.testing.assert_close(model.opd_projector_grad_sum, expected, rtol=2e-05, atol=1e-07)
    assert model.opd_projector_grad_weight.item() > 0
    p = torch.nn.Parameter(old_a.clone())
    model.opd_projector = p
    gradient = model.opd_projector_grad_sum / model.opd_projector_grad_weight
    optimizer = torch.optim.SGD([p], lr=0.1)
    p.grad = gradient.clone()
    optimizer.step()
    assert not torch.equal(p, old_a)

def test_training_checkpoint_restores_rank_local_pending_projector_gradient(tmp_path):
    """Exercise real checkpoint functions without importing the training CLI."""

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.target_model = torch.nn.Linear(8, 8)
            self.draft_model = torch.nn.Linear(8, 8)
            self.draft_model.register_parameter('opd_projector', torch.nn.Parameter(torch.eye(8)))
            self.draft_model.register_buffer('opd_projector_grad_sum', torch.ones(8, 8), persistent=False)
            self.draft_model.register_buffer('opd_projector_grad_weight', torch.ones(1), persistent=False)
            self._training_method = 'opd_reflex'

        @property
        def opd_projector(self):
            return self.draft_model.opd_projector

        @property
        def opd_projector_grad_sum(self):
            return self.draft_model.opd_projector_grad_sum

        @property
        def opd_projector_grad_weight(self):
            return self.draft_model.opd_projector_grad_weight

        def load_opd_projector(self, value):
            FastGRPOModel.load_opd_projector(self, value)
    rank = [0]

    def gather(rows, local):
        rows[0] = copy.deepcopy(local)
        rows[1] = copy.deepcopy(local)
        rows[1]['rank'] = 1
        rows[1]['opd_projector_pending_sum'].fill_(7)
        rows[1]['opd_projector_pending_weight'].fill_(3)
    scope = dict(torch=torch, Path=Path, dist=SimpleNamespace(is_initialized=lambda : True, get_world_size=lambda : 2, get_rank=lambda : rank[0], all_gather_object=gather), capture_rng_state=lambda : None, restore_rng_state=lambda x: None, _atomic_torch_save=lambda state, path: torch.save(state, path), _prune_checkpoints=lambda *a: None, get_peft_model_state_dict=None, set_peft_model_state_dict=None)
    from helper.opd_optimizer import load_draft_optimizer
    scope['load_draft_optimizer'] = load_draft_optimizer
    source = ast.parse((Path(__file__).parents[1] / 'tests/references/reference_checkpoint.py').read_text())
    names = {'_target_lora_state_dict', '_load_target_lora_state_dict', '_gradient_state', '_restore_gradient_state', 'save_training_checkpoint', 'load_training_checkpoint'}
    nodes = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'checkpoint-functions', 'exec'), scope)
    model = Model()
    target = torch.optim.SGD(model.target_model.parameters(), lr=0.01)
    draft = torch.optim.SGD(model.draft_model.parameters(), lr=0.01)
    checkpoint = scope['save_training_checkpoint'](tmp_path, model=model, optimizer_target=target, optimizer_draft=draft, epoch=0, next_batch=1, step=1, used_items=1, draft_step=0, draft_accumulated_step=1, batch_data={}, keep_last=1, cumulative_elapsed_time_s=1.0)
    rank[0] = 1
    model.opd_projector_grad_sum.zero_()
    model.opd_projector_grad_weight.zero_()
    scope['load_training_checkpoint'](checkpoint, model=model, optimizer_target=target, optimizer_draft=draft)
    assert torch.equal(model.opd_projector_grad_sum, torch.full((8, 8), 7.0))
    assert torch.equal(model.opd_projector_grad_weight, torch.tensor([3.0]))

def test_fastgrpo_checkpoint_resume_with_none_projector_property(tmp_path):

    class Model(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.target_model = torch.nn.Linear(3, 4)
            self.draft_model = torch.nn.Linear(3, 4)
            self._training_method = 'fastgrpo'

        @property
        def opd_projector(self):
            return None
    from helper.opd_optimizer import load_draft_optimizer
    scope = dict(torch=torch, Path=Path, dist=SimpleNamespace(is_initialized=lambda : False), capture_rng_state=lambda : None, restore_rng_state=lambda state: None, _atomic_torch_save=lambda state, path: torch.save(state, path), _prune_checkpoints=lambda *args: None, get_peft_model_state_dict=None, set_peft_model_state_dict=None, load_draft_optimizer=load_draft_optimizer)
    names = {'_target_lora_state_dict', '_load_target_lora_state_dict', '_gradient_state', '_restore_gradient_state', 'save_training_checkpoint', 'load_training_checkpoint'}
    nodes = [n for n in ast.parse((Path(__file__).parents[1] / 'tests/references/reference_checkpoint.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-checkpoint-functions', 'exec'), scope)
    model = Model()
    target = torch.optim.AdamW(model.target_model.parameters(), lr=1e-06)
    draft = torch.optim.AdamW(model.draft_model.parameters(), lr=0.0001)
    expected = {name: p.detach().clone() for (name, p) in model.named_parameters()}
    checkpoint = scope['save_training_checkpoint'](tmp_path, model=model, optimizer_target=target, optimizer_draft=draft, epoch=0, next_batch=1, step=1, used_items=8, draft_step=1, draft_accumulated_step=1, batch_data={}, keep_last=1, cumulative_elapsed_time_s=1.0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(10)
    restored = scope['load_training_checkpoint'](checkpoint, model=model, optimizer_target=target, optimizer_draft=draft)
    assert restored['method'] == 'fastgrpo' and model.opd_projector is None
    for (name, p) in model.named_parameters():
        assert torch.equal(p, expected[name])

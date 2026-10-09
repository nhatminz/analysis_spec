import csv
import copy
from types import SimpleNamespace
import pytest
import torch
from helper.opd_sampling import sample_target_with_metadata
from helper.opd_static_cache import OPDStaticCache
from helper.rollout_metrics import RolloutMetricsWriter
from helper.opd_optimizer import draft_optimizer, load_draft_optimizer
from helper.tree_verification import PackedTree
from test_opd_reflex import state
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device', DEVICES)
def test_static_append_crop_repeat_compact_without_prefix_copy(device):
    c = OPDStaticCache(64)
    parts = []
    for length in (5, 3, 7):
        x = torch.randn(2, 3, length, 4, device=device)
        parts.append(x)
        (k, v) = c.update(x, x + 1, 0)
        if len(parts) == 1:
            ptr = k.untyped_storage().data_ptr()
        assert k.untyped_storage().data_ptr() == ptr
        assert torch.equal(k, torch.cat(parts, dim=-2))
    c.crop(6)
    snapshot = c[0][0].clone()
    c.batch_repeat_interleave(2)
    assert torch.equal(c[0][0], snapshot.repeat_interleave(2, 0))
    c.batch_select_indices(torch.tensor([1, 3], device=device))
    assert torch.equal(c[0][0], snapshot)
    assert torch.equal(c[0][1], snapshot + 1)
    c.crop(0)
    c.update(torch.ones(2, 3, 1, 4, device=device), torch.ones(2, 3, 1, 4, device=device), 0)
    assert c.get_seq_length() == 1

def test_iteration_csv_zero_reward_weighted_aal_resume_and_idempotence(tmp_path):
    path = tmp_path / 'rollout_timing.csv'
    w = RolloutMetricsWriter(path, 'fastgrpo', flush_interval=10)
    used = 0
    for (i, (acc, rounds, eligible)) in enumerate(((9, 3, 0), (2, 2, 2), (30, 5, 0))):
        w.begin(1, i, 8, used)
        used += eligible
        w.finish(dict(total_acc_length=acc, total_decoded_token_num=rounds, total_time_cost=2.0, response_generated_tokens=[4, 5]), grpo_step=0, used_items=used, wall_time_s=(i + 1) * 3)
        w.finish(None, grpo_step=0, used_items=used, wall_time_s=0)
        if i == 1:
            checkpoint = copy.deepcopy(w.state)
    w.close()
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 3 and float(rows[-1]['cumulative_aal']) == 4.1
    assert [int(r['eligible_prompts']) for r in rows] == [0, 2, 0]
    w = RolloutMetricsWriter(path, 'fastgrpo', state=checkpoint)
    w.begin(2, 0, 8, 2)
    w.finish(None, grpo_step=1, used_items=2, wall_time_s=10.0)
    w.close()
    rows = list(csv.DictReader(path.open()))
    assert [r['global_iter'] for r in rows] == ['1', '2', '3']
    assert float(rows[-1]['cumulative_aal']) == 11 / 5
    assert all((float(r['iter_opd_kl']) == 0 for r in rows))

def test_projector_lr_migrates_optimizer_moments():
    model = torch.nn.Linear(3, 4)
    model.register_parameter('opd_projector', torch.nn.Parameter(torch.ones(3, 2)))
    old = draft_optimizer(model, 1e-05)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    old.step()
    saved = copy.deepcopy(old.state_dict())
    new = draft_optimizer(model, 1e-05, 0.0003)
    load_draft_optimizer(new, saved, model)
    assert new.param_groups[1]['lr'] == 0.0003
    for p in model.parameters():
        assert torch.equal(old.state[p]['exp_avg'], new.state[p]['exp_avg'])
        assert torch.equal(old.state[p]['step'], new.state[p]['step'])
    resumed = draft_optimizer(model, 1e-05)
    load_draft_optimizer(resumed, new.state_dict(), model)
    assert resumed.param_groups[1]['lr'] == 0.0003
    assert torch.equal(resumed.state[model.opd_projector]['exp_avg'], old.state[model.opd_projector]['exp_avg'])

@pytest.mark.skipif(not torch.cuda.is_available(), reason='greedy teacher CUDA')
def test_greedy_full_vocabulary_never_scans_teacher(monkeypatch):
    from helper import opd_reflex_kernels as kernels
    (s, model, _) = state('cuda', v=37)
    mapping = torch.arange(37, device='cuda')
    inverse = torch.empty_like(mapping)
    inverse[mapping] = torch.arange(37, device='cuda')
    model.opd_full_vocab_inverse = inverse
    s.start(model, 3, mapping, 32, max_contexts=8, max_nodes=24, max_path=5, max_proposal_contexts=4)
    h = torch.randn(3, 1, 32, device='cuda')
    s.propose(model.lm_head(h), h, 8, mapping, root=True)
    root = torch.full((3, 1), -1, device='cuda', dtype=torch.long)

    def forbidden(*a, **k):
        raise AssertionError('unnecessary greedy teacher scan')
    monkeypatch.setattr(kernels, 'teacher', forbidden)
    targets = torch.tensor([[1], [17], [21]], device='cuda')
    s.feedback(PackedTree(root, root.clone(), torch.zeros_like(root), 0), SimpleNamespace(packed_indices=torch.zeros_like(root)), targets, greedy=True)
    assert torch.equal(s.teacher_ids[:48].view(3, 16)[:, 0], inverse[targets[:, 0]])
    assert not s.teacher_p[:48].view(3, 16)[:, 1:].any()

def test_iteration_telemetry_does_not_retain_gpu_history():
    data = {'all_target_hidden_states': torch.empty(2, 3, 8), 'response_generated_tokens': [2, 3], 'opd_selected_states': 2.0, 'opd_extra_tensor': torch.empty(4), 'total_acc_length': 5}
    captured = RolloutMetricsWriter.capture(data)
    assert set(captured) == {'response_generated_tokens', 'opd_selected_states', 'total_acc_length'}

@pytest.mark.skipif(not torch.cuda.is_available(), reason='sparse CUDA scratch capacity')
def test_sparse_scratch_never_reserves_full_context_vocabulary(monkeypatch):
    from test_opd_reflex import seed
    monkeypatch.setenv('OPD_PROPOSAL_MODE', 'sparse')
    (s, model, mapping) = state('cuda', v=65537)
    seed(s, torch.arange(16, device='cuda'), torch.randn(16, 8, device='cuda'))
    h = torch.randn(3, 4, 32, device='cuda')
    s.propose(model.lm_head(h), h, 8, mapping)
    assert s.sparse_capacity == 16 + 2 * 24 * 16
    assert s.sparse_scores.numel() == 3 * 4 * s.sparse_capacity
    assert s.score_workspace.numel() == 1
    before = s.sparse_scores.untyped_storage().data_ptr()
    s.propose(model.lm_head(h[:1, :1]), h[:1, :1], 8, mapping)
    assert s.sparse_scores.untyped_storage().data_ptr() == before

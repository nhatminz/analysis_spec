import ast
import csv
import weakref
from pathlib import Path
import pytest
import torch
from helper.opd_static_cache import OPDStaticCache
from helper.opd_attention import AttentionWorkspace
from helper.opd_sampling import sample_target_with_metadata
from helper.rollout_metrics import FIELDS, KV_FIELDS, RolloutMetricsWriter
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

def test_cache_does_not_retain_pools_in_owner_cycle():
    cache = OPDStaticCache(8)
    x = torch.zeros(2, 3, 4, 5)
    cache.update(x, x, 0)
    cache_ref = weakref.ref(cache)
    pool_ref = weakref.ref(cache.layers[0].key_pool)
    del cache
    assert cache_ref() is None and pool_ref() is None

def test_old_iteration_csv_resume_migrates_header_and_logs_exact_kv_counters(tmp_path):
    path = tmp_path / 'rollout_timing.csv'
    fields = [name for name in FIELDS if name not in KV_FIELDS]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(global_iter=1, method='opd_reflex', grpo_step=0))
    state = dict(global_iter=1, accepted=4, rounds=2, tokens=5, generation=1, accepted_draft=2, proposed=4)
    writer = RolloutMetricsWriter(path, 'opd_reflex', state=state)
    writer.begin(1, 1, 8, 0)
    writer.finish(dict(total_acc_length=9, total_decoded_token_num=3, opd_host_syncs=2, opd_host_syncs_per_round=1, opd_target_kv_cache_bytes=100, opd_draft_kv_cache_bytes=30, opd_target_full_kv_reallocations=1, opd_draft_full_kv_reallocations=2, opd_target_full_history_copies=4, opd_draft_full_history_copies=5, opd_target_kv_rows_moved=3, opd_draft_kv_rows_moved=3, opd_target_full_history_copy_bytes=80, opd_draft_full_history_copy_bytes=90), grpo_step=1, used_items=8, wall_time_s=2)
    writer.close()
    assert path.with_suffix('.pre_resume.csv').is_file()
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['global_iter'] for row in rows] == ['1', '2']
    assert list(rows[1]) == list(FIELDS) and rows[0]['iter_host_syncs'] == ''
    assert float(rows[1]['iter_aal']) == 3
    for (field, value) in zip(KV_FIELDS, [2, 1, 130, 3, 9, 170, 3, 0]):
        assert float(rows[1][field]) == value

@pytest.mark.parametrize('device', DEVICES)
def test_geometric_growth_and_no_pool_replacement_on_repeat_select(device):
    c = OPDStaticCache(8, batch_capacity=8, chunk_size=8)
    full = torch.randn(2, 3, 39, 5, device=device)
    values = full + 1
    for (start, end) in ((0, 5), (5, 8), (8, 9), (9, 17), (17, 39)):
        (k, v) = c.update(full[..., start:end, :], values[..., start:end, :], 0)
        assert torch.equal(k, full[..., :end, :]) and torch.equal(v, values[..., :end, :])
        assert c.layers[0].capacity <= 2 * max(8, end)
    stats = c.statistics()
    assert stats['full_kv_reallocations'] == 3
    assert stats['full_history_copies'] == 3
    ptr = c[0][0].untyped_storage().data_ptr()
    c.batch_repeat_interleave(4)
    oracle = full.repeat_interleave(4, 0)
    v_oracle = values.repeat_interleave(4, 0)
    assert torch.equal(c[0][0], oracle)
    for ids in ([7, 0, 4, 2, 6], [4, 0, 0, 3], [3, 1]):
        ids = torch.tensor(ids, device=device)
        oracle = oracle.index_select(0, ids)
        v_oracle = v_oracle.index_select(0, ids)
        c.batch_select_indices(ids)
        assert torch.equal(c[0][0], oracle) and torch.equal(c[0][1], v_oracle)
        assert c[0][0].untyped_storage().data_ptr() == ptr
        assert c.statistics()['full_kv_reallocations'] == stats['full_kv_reallocations']
    c.crop(13)
    suffix = torch.randn(2, 3, 6, 5, device=device)
    c.update(suffix, suffix + 1, 0)
    assert torch.equal(c[0][0], torch.cat((oracle[..., :13, :], suffix), -2))
    assert c[0][0].untyped_storage().data_ptr() == ptr

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA alias/race stress')
@pytest.mark.parametrize('batch', [16, 32, 64])
def test_compaction_race_free_arbitrary_permutation_and_duplicates(batch):
    c = OPDStaticCache(256, batch_capacity=batch)
    k = torch.randn(batch, 3, 517, 37, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    c.update(k, v, 0)
    ptr = c.layers[0].key_pool.data_ptr()
    for _ in range(5):
        ids = torch.randperm(batch, device='cuda')
        ids[-4:] = ids[:4]
        k = k.index_select(0, ids)
        v = v.index_select(0, ids)
        c.batch_select_indices(ids)
        assert torch.equal(c[0][0], k) and torch.equal(c[0][1], v)
        assert c.layers[0].key_pool.data_ptr() == ptr
    assert c.statistics()['full_kv_reallocations'] == 0
    assert c.statistics()['kv_compaction_workspace_bytes'] <= 2 * c.statistics()['kv_cache_bytes']

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_reused_attention_mask_matches_original_and_positions_own_storage(device, dtype):
    w = AttentionWorkspace(device)
    for (past, b, q) in ((1, 3, 6), (7, 3, 4), (1, 2, 3), (257, 2, 6), (17, 1, 2)):
        pad = torch.zeros(b, past + q, device=device, dtype=torch.bool)
        pad[:, 1:3] = True
        expected = torch.triu(torch.full((q, past + q), torch.finfo(dtype).min, device=device, dtype=dtype), diagonal=past + 1)
        expected = expected[None, None].repeat(b, 1, 1, 1).masked_fill(pad[:, None, None], torch.finfo(dtype).min)
        actual = w.causal('mask', past, q, b, dtype, pad)
        assert torch.equal(actual, expected)
        assert torch.equal(w.positions('position', past, q), torch.arange(past, past + q, device=device))
    allocations = w.allocations
    for _ in range(4):
        w.causal('mask', 17, 2, 1, dtype, pad)
    assert w.allocations == allocations

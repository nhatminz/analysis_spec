import ast
import weakref
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from helper.opd_static_cache import OPDStaticCache, swap_remove_plan, persistent_cache
from helper.opd_sampling import sample_target_with_metadata
from helper.opd_reflex import OPDReflex
from helper.tree_verification import PackedTree
from test_opd_reflex import state, seed
ROOT = Path(__file__).resolve().parents[1]
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('finished', [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 0, 0, 1, 0], [1, 0, 1, 0, 1, 0], [1, 1, 1]])
def test_swap_remove_only_moves_disjoint_live_tail_rows(device, finished):
    b = len(finished)
    k = torch.randn(b, 3, 517, 7, device=device)
    c = OPDStaticCache(256, batch_capacity=b)
    c.update(k, k + 1, 0)
    (keep, sources, destinations) = swap_remove_plan(finished)
    old = c.statistics()
    ptr = c.layers[0].key_pool.data_ptr()
    c.swap_remove(len(keep), torch.tensor(sources, device=device, dtype=torch.long), torch.tensor(destinations, device=device, dtype=torch.long))
    expected = k.index_select(0, torch.tensor(keep, device=device, dtype=torch.long))
    assert torch.equal(c[0][0], expected) and torch.equal(c[0][1], expected + 1)
    assert c.layers[0].key_pool.data_ptr() == ptr
    stats = c.statistics()
    copied = len(sources) * 3 * 517 * 7 * 4 * 2
    assert stats['kv_rows_moved'] == len(sources)
    assert stats['kv_history_copy_bytes'] - old['kv_history_copy_bytes'] == copied
    assert set(sources).isdisjoint(destinations)
    assert all((source >= len(keep) for source in sources))
    if finished == [0, 0, 0, 1]:
        assert copied == 0

@pytest.mark.parametrize('device', DEVICES)
def test_persistent_pools_reuse_high_water_mark_without_stale_history(device):
    model = SimpleNamespace()
    pointers = []
    for (iteration, (b, length)) in enumerate(((2, 601), (1, 17), (2, 300), (2, 550))):
        c = persistent_cache(model, '_opd_target_kv_pool', b, 4, device, torch.float32)
        assert c.get_seq_length() == 0 and (not c)
        k = torch.full((b, 2, length, 7), float(iteration + 1), device=device)
        c.update(k, k + 10, 0)
        c.batch_repeat_interleave(4)
        assert torch.equal(c[0][0], k.repeat_interleave(4, 0))
        pointers.append(c.layers[0].key_pool.data_ptr())
        if iteration:
            assert c.statistics()['full_kv_reallocations'] == 0
        c.end_rollout()
        assert c.get_seq_length() == 0 and c[0][0].shape[0] == 0
    assert len(set(pointers)) == 1
    c.end_rollout(max_retained_tokens=256)
    assert not c.layers and (not c._scratch)

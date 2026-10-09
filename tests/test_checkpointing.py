import random

import numpy as np
import torch

from helper.checkpointing import capture_optimizer_and_rng, restore_optimizer_and_rng


def test_checkpoint_restores_optimizer_and_all_rng_states():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    payload = capture_optimizer_and_rng(optimizer)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    optimizer.param_groups[0]["lr"] = 9.0
    random.random(); np.random.rand(); torch.rand(3)
    restore_optimizer_and_rng(payload, optimizer)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])

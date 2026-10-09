"""Small checkpoint primitives shared by runtime code and CPU unit tests."""

from __future__ import annotations

import random

import numpy as np
import torch


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def capture_optimizer_and_rng(optimizer):
    return {"optimizer": optimizer.state_dict(), "rng": capture_rng_state()}


def restore_optimizer_and_rng(payload, optimizer):
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng_state(payload["rng"])

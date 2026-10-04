"""Paired shadow evaluation and the absolute target-step analysis schedule."""

from __future__ import annotations


def is_analysis_boundary(step, explicit, interval, total_steps):
    return (step > 0 and (total_steps <= 0 or step < total_steps)
            and (step in explicit or (interval > 0 and step % interval == 0)))


def branch_spec(branch):
    """The Reflex branch always uses the stale persistent draft, never fresh."""
    if branch not in {"stale", "fresh", "reflex"}:
        raise ValueError(f"unknown policy-lag branch: {branch}")
    return ("fresh" if branch == "fresh" else "stale",
            "active" if branch == "reflex" else "off")


def paired_shadow_rollout(*, branch, branch_state, stale_state, rng_state,
                          expected_draft_id, digest, load_draft, read_draft,
                          persistent_identity, capture_rng, restore_rng, rollout):
    """Run a shadow with the paired initial RNG; always restore the live stale.

    Callbacks keep this helper usable by CPU toy tests without importing the
    top-level GPU training script. The persistent fingerprint includes target,
    draft optimizer and the real GRPO buffer, but deliberately excludes RNG:
    generation consumes RNG which we restore, not persist.
    """
    _, mode = branch_spec(branch)
    before = persistent_identity()
    if digest(branch_state) != expected_draft_id:
        raise RuntimeError(f"{branch} was given the wrong persistent draft")
    if branch == "reflex" and digest(branch_state) != digest(stale_state):
        raise RuntimeError("Reflex must evaluate phi_stale")
    try:
        load_draft(branch_state)
        restore_rng(rng_state)
        if digest(capture_rng()) != digest(rng_state):
            raise RuntimeError("shadow RNG was not restored to the captured initial state")
        result = rollout(mode)
        if digest(read_draft()) != expected_draft_id:
            raise RuntimeError(f"{branch} shadow modified persistent draft weights")
        if persistent_identity() != before:
            raise RuntimeError(f"{branch} shadow modified target/optimizer/GRPO state")
        return result
    finally:
        load_draft(stale_state)
        restore_rng(rng_state)
        if digest(read_draft()) != digest(stale_state):
            raise RuntimeError("live phi_stale was not restored after shadow rollout")
        if digest(capture_rng()) != digest(rng_state):
            raise RuntimeError("live RNG was not restored after shadow rollout")

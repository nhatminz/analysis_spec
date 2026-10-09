def _target_lora_state_dict(target_model):
    if get_peft_model_state_dict is not None:
        return get_peft_model_state_dict(target_model)
    return target_model.state_dict()

def _load_target_lora_state_dict(target_model, state_dict):
    if set_peft_model_state_dict is not None:
        set_peft_model_state_dict(target_model, state_dict)
    else:
        target_model.load_state_dict(state_dict, strict=False)

def _gradient_state(module):
    return {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }

def _restore_gradient_state(module, state):
    parameters = dict(module.named_parameters())
    for name, value in (state or {}).items():
        if name not in parameters:
            raise ValueError(f"checkpoint gradient parameter is missing: {name}")
        parameters[name].grad = value.to(parameters[name].device, parameters[name].dtype)

def save_training_checkpoint(
    checkpoint_dir,
    *,
    model,
    optimizer_target,
    optimizer_draft,
    epoch,
    next_batch,
    step,
    used_items,
    draft_step,
    draft_accumulated_step,
    batch_data,
    keep_last,
    cumulative_elapsed_time_s,
):
    current_rank = dist.get_rank() if dist.is_initialized() else 0
    current_world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_state = {
        "rank": int(current_rank),
        "used_items": int(used_items),
        "batch_data": batch_data,
        "target_gradients": _gradient_state(model.target_model),
        "draft_gradients": _gradient_state(model.draft_model),
        "rng": capture_rng_state(),
        "cumulative_elapsed_time_s": float(cumulative_elapsed_time_s),
    }
    # Pending analytical gradients are rank-local until the existing optimizer
    # boundary. Never restore rank 0's pending feedback into every rollout worker.
    for key, attribute in (
        ('opd_projector_pending_sum', 'opd_projector_grad_sum'),
        ('opd_projector_pending_weight', 'opd_projector_grad_weight'),
    ):
        value = getattr(model, attribute, None)
        local_state[key] = None if value is None else value.detach().cpu().clone()
    if dist.is_initialized():
        rank_states = [None] * current_world_size
        dist.all_gather_object(rank_states, local_state)
    else:
        rank_states = [local_state]
    if current_rank != 0:
        return None
    checkpoint_dir = Path(checkpoint_dir)
    state = {
        "format": "opd_fastgrpo_checkpoint_v5",
        "world_size": int(current_world_size),
        "rank_states": rank_states,
        "cumulative_elapsed_time_s": max(
            float(item["cumulative_elapsed_time_s"]) for item in rank_states
        ),
        "epoch": int(epoch),
        "next_batch": int(next_batch),
        "step": int(step),
        "used_items": int(used_items),
        "draft_step": int(draft_step),
        "draft_accumulated_step": int(draft_accumulated_step),
        "target_lora": _target_lora_state_dict(model.target_model),
        "draft_model": model.draft_model.state_dict(),
        "opd_projector": getattr(model,"opd_projector",None),
        "opd_projector_pending_sum": getattr(model,'opd_projector_grad_sum',None),
        "opd_projector_pending_weight": getattr(model,'opd_projector_grad_weight',None),
        "method": getattr(model,"_training_method","fastgrpo"),
        "optimizer_target": optimizer_target.state_dict(),
        "optimizer_draft": optimizer_draft.state_dict(),
        "scheduler_target": None,
        "scheduler_draft": None,
    }
    checkpoint_path = checkpoint_dir / f"step{int(step)}_epoch{int(epoch) + 1}_batch{int(next_batch)}.pt"
    _atomic_torch_save(state, checkpoint_path)
    _atomic_torch_save(state, checkpoint_dir / "latest.pt")
    _prune_checkpoints(checkpoint_dir, keep_last)
    print(f"Saved FastGRPO checkpoint: {checkpoint_path}")
    return checkpoint_path

def load_training_checkpoint(path, *, model, optimizer_target, optimizer_draft):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    current_world_size = dist.get_world_size() if dist.is_initialized() else 1
    current_rank = dist.get_rank() if dist.is_initialized() else 0
    saved_world_size = int(checkpoint.get("world_size", 1))
    if saved_world_size != current_world_size:
        raise RuntimeError(
            "checkpoint world-size mismatch: "
            f"saved={saved_world_size}, current={current_world_size}; "
            "resume with the same number of ranks"
        )
    rank_states = checkpoint.get("rank_states")
    if rank_states is not None:
        if len(rank_states) != saved_world_size:
            raise RuntimeError("checkpoint rank_states does not match saved world_size")
        local_state = rank_states[current_rank]
        if int(local_state.get("rank", current_rank)) != current_rank:
            raise RuntimeError(f"checkpoint has no state for rank {current_rank}")
    else:
        local_state = {
            "used_items": checkpoint.get("used_items", 0),
            "batch_data": checkpoint.get("batch_data", {}),
            "target_gradients": checkpoint.get("target_gradients"),
            "draft_gradients": checkpoint.get("draft_gradients"),
            "rng": checkpoint.get("all_rng_states"),
        }
    if checkpoint.get("method","fastgrpo")!=getattr(model,"_training_method","fastgrpo"):
        raise ValueError("resume method mismatch; start a new run for replacement OPD")
    if getattr(model, 'opd_projector', None) is not None and 'opd_projector' not in checkpoint['draft_model']:
        raise ValueError(
            'resume checkpoint predates the learned draft projector; start a new '
            'run using its draft weights as initialization, not optimizer resume'
        )
    model.draft_model.load_state_dict(checkpoint["draft_model"])
    if checkpoint.get("opd_projector") is not None:model.load_opd_projector(checkpoint["opd_projector"])
    for key, attribute in (
        ('opd_projector_pending_sum', 'opd_projector_grad_sum'),
        ('opd_projector_pending_weight', 'opd_projector_grad_weight'),
    ):
        value = local_state.get(key, checkpoint.get(key))
        if value is not None:
            getattr(model, attribute).copy_(value)
    _load_target_lora_state_dict(model.target_model, checkpoint["target_lora"])
    optimizer_target.load_state_dict(checkpoint["optimizer_target"])
    load_draft_optimizer(optimizer_draft,checkpoint["optimizer_draft"],model.draft_model)
    _restore_gradient_state(model.target_model, local_state.get("target_gradients"))
    _restore_gradient_state(model.draft_model, local_state.get("draft_gradients"))
    if local_state.get("rng") is not None:
        restore_rng_state(local_state["rng"])
    checkpoint["used_items"] = int(local_state.get("used_items", 0))
    checkpoint["batch_data"] = local_state.get("batch_data", {})
    return checkpoint

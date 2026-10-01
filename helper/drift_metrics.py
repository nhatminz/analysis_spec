"""Memory-bounded draft/teacher drift metrics.

This module is intentionally local to FastGRPO.  Policy-lag analysis must not
depend on an optional sibling ``flashgrpo`` checkout just to compute logging
metrics.
"""

import torch


def _acceptance_aligned_union_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: int = 64,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return TV and forward KL on a bidirectional top-k union plus tail."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have identical shapes")
    temperature = max(float(temperature), 1e-4)
    student_scaled = student_logits.float() / temperature
    teacher_scaled = teacher_logits.float() / temperature
    k = min(max(1, int(topk)), int(student_scaled.shape[-1]))

    teacher_ids = torch.topk(teacher_scaled, k=k, dim=-1).indices
    student_ids = torch.topk(student_scaled, k=k, dim=-1).indices
    support_ids = torch.cat((teacher_ids, student_ids), dim=-1).sort(dim=-1).values
    unique = torch.ones_like(support_ids, dtype=torch.bool)
    unique[..., 1:] = support_ids[..., 1:] != support_ids[..., :-1]

    teacher_log_z = torch.logsumexp(teacher_scaled, dim=-1, keepdim=True)
    student_log_z = torch.logsumexp(student_scaled, dim=-1, keepdim=True)
    teacher_logp = torch.gather(teacher_scaled, -1, support_ids) - teacher_log_z
    student_logp = torch.gather(student_scaled, -1, support_ids) - student_log_z
    teacher_prob = teacher_logp.exp().masked_fill(~unique, 0.0)
    student_prob = student_logp.exp().masked_fill(~unique, 0.0)
    teacher_tail = (1.0 - teacher_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)
    student_tail = (1.0 - student_prob.sum(dim=-1)).clamp(min=1e-8, max=1.0)

    tv = 0.5 * (
        (teacher_prob - student_prob).abs().sum(dim=-1)
        + (teacher_tail - student_tail).abs()
    )
    kl = (
        teacher_prob * (teacher_logp - student_logp).masked_fill(~unique, 0.0)
    ).sum(dim=-1) + teacher_tail * (
        torch.log(teacher_tail) - torch.log(student_tail)
    )
    valid = valid_mask.to(device=tv.device, dtype=torch.float32)
    denominator = valid.sum().clamp_min(1.0)
    return (
        (tv * valid).sum() / denominator,
        (kl.clamp_min(0.0) * valid).sum() / denominator,
    )


@torch.no_grad()
def acceptance_aligned_union_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    topk: int = 64,
    temperature: float = 1.0,
    row_chunk_size: int = 32,
) -> tuple[float, float, int]:
    """Measure sparse TV/KL without materializing every vocabulary row."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have identical shapes")
    if student_logits.shape[:-1] != valid_mask.shape:
        raise ValueError("valid_mask must match the non-vocabulary logit dimensions")

    valid_coords = valid_mask.to(
        device=student_logits.device, dtype=torch.bool
    ).nonzero(as_tuple=False)
    count = int(valid_coords.shape[0])
    if count == 0:
        return 0.0, 0.0, 0

    chunk_size = max(1, int(row_chunk_size))
    tv_sum = 0.0
    kl_sum = 0.0
    for start in range(0, count, chunk_size):
        coords = valid_coords[start : start + chunk_size]
        index = tuple(coords[:, dim] for dim in range(int(coords.shape[1])))
        chunk_student = student_logits[index]
        chunk_teacher = teacher_logits[index]
        chunk_valid = torch.ones(
            (int(coords.shape[0]),),
            dtype=torch.bool,
            device=student_logits.device,
        )
        tv, kl = _acceptance_aligned_union_loss(
            chunk_student,
            chunk_teacher,
            chunk_valid,
            topk=topk,
            temperature=temperature,
        )
        chunk_count = int(coords.shape[0])
        tv_sum += float(tv.detach().cpu()) * chunk_count
        kl_sum += float(kl.detach().cpu()) * chunk_count

    return tv_sum / count, kl_sum / count, count


__all__ = ["acceptance_aligned_union_metrics"]

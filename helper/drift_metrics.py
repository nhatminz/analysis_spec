"""Optional legacy draft-drift logging without a dependency on MEDUSA code."""

import torch


@torch.no_grad()
def sparse_union_metrics(student_logits, teacher_logits, valid_mask, *, topk=64,
                         temperature=1.0, row_chunk_size=32):
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have identical shapes")
    coords = valid_mask.to(student_logits.device, torch.bool).nonzero(as_tuple=False)
    count = int(coords.shape[0])
    if count == 0:
        return 0.0, 0.0, 0
    tv_sum = kl_sum = 0.0
    temp = max(float(temperature), 1e-4)
    for start in range(0, count, max(1, int(row_chunk_size))):
        cur = coords[start:start + max(1, int(row_chunk_size))]
        index = tuple(cur[:, dim] for dim in range(cur.shape[1]))
        student = student_logits[index].float() / temp
        teacher = teacher_logits[index].float() / temp
        k = min(max(1, int(topk)), student.shape[-1])
        ids = torch.cat((teacher.topk(k, -1).indices, student.topk(k, -1).indices), -1)
        ids = ids.sort(-1).values
        unique = torch.ones_like(ids, dtype=torch.bool)
        unique[..., 1:] = ids[..., 1:] != ids[..., :-1]
        t_logp = teacher.gather(-1, ids) - teacher.logsumexp(-1, keepdim=True)
        s_logp = student.gather(-1, ids) - student.logsumexp(-1, keepdim=True)
        tp = t_logp.exp().masked_fill(~unique, 0)
        sp = s_logp.exp().masked_fill(~unique, 0)
        tt = (1 - tp.sum(-1)).clamp(1e-8, 1)
        st = (1 - sp.sum(-1)).clamp(1e-8, 1)
        tv = 0.5 * ((tp - sp).abs().sum(-1) + (tt - st).abs())
        kl = (tp * (t_logp - s_logp).masked_fill(~unique, 0)).sum(-1)
        kl = (kl + tt * (tt.log() - st.log())).clamp_min(0)
        tv_sum += float(tv.sum().cpu())
        kl_sum += float(kl.sum().cpu())
    return tv_sum / count, kl_sum / count, count

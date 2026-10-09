"""Reusable OPD tree/mask/path workspaces; baseline uses upstream storage."""
import torch


def allocate_tree_buffers(runtime,alloc,batch,max_contexts,max_path,max_proposal_contexts):
    """Allocate reusable tree storage for the optimized OPD rollout."""
    runtime.path_workspace=[alloc((batch,max_path),torch.long) for _ in range(3)]+[alloc(batch,torch.long)]
    runtime.padded_path_workspace=[alloc((batch,max_path),torch.bool if i==2 else torch.long) for i in range(3)]+[alloc((batch,1),torch.long)]
    runtime.scheduling_packet=alloc((batch,max_path+4),torch.long)
    full_nodes=max_proposal_contexts*max_contexts
    runtime.tree_buffers={name:alloc((batch,full_nodes),torch.float32 if name=='confidence' else torch.long)
        for name in ('parents','contexts','tokens','positions','confidence')}
    runtime.tree_arange=torch.arange(full_nodes+1,device=runtime.attention_workspace.device,dtype=torch.long)
    runtime.tree_seen=[alloc((batch,max_proposal_contexts,max_path),torch.long) for _ in range(2)]
    runtime.tree_positions=alloc((batch,max_proposal_contexts),torch.long)
    runtime.tree_branch_confidence=alloc((batch,max_proposal_contexts,max_proposal_contexts))
    runtime.tree_top_values=alloc((batch,max_proposal_contexts))
    runtime.tree_top_indices=alloc((batch,max_proposal_contexts),torch.long)
    runtime.pack_workspace=[alloc(batch*(full_nodes+1),torch.long) for _ in range(4)]
    runtime.tree_root_confidences=alloc((batch,max_proposal_contexts))



"""Shared early-EOS bookkeeping, without inventing verification rounds."""
import torch


def prefill_eos_output(input_ids,features,padding_positions,repeats,eos,return_history,duration):
    repeats=max(1,repeats or 1);count=input_ids.shape[0]*repeats
    states=[];ids=[]
    if return_history:
        for row,padding in enumerate(padding_positions):
            keep=[i for i in range(input_ids.shape[1]) if i not in padding]
            prompt=input_ids[row,keep]
            draft=torch.cat((prompt[1:],prompt.new_tensor([eos])))
            hidden=features[row,keep]
            for _ in range(repeats):states.append(hidden.clone());ids.append(draft.clone())
    return dict(generated_token_ids=[[eos] for _ in range(count)],max_sequence_length=1,
                all_draft_input_states=states if return_history else None,all_draft_input_ids=ids if return_history else None,
                response_accepted_length_sum=[0]*count,response_verification_rounds=[0]*count,
                response_generated_tokens=[1]*count,total_acc_length=0,total_decoded_token_num=0,total_acc=0.,
                total_accepted_draft_tokens=0,total_proposed_draft_tokens=0,draft_acceptance_rate=0.,
                verification_batches=0,batch_verification_rounds=0,active_response_rounds=0,verified_tree_nodes=0,
                total_time_cost=duration,target_time_cost=duration,draft_time_cost=0.,check_time_cost=0.,
                prefill_time_cost=duration,post_time_cost=0.)

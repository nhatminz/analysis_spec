import pandas as pd
from transformers import AutoTokenizer, AutoProcessor,AutoConfig,AutoModelForCausalLM
import torch
import datasets

from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import time
from pathlib import Path

from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
import datasets
from transformers import get_cosine_schedule_with_warmup
from transformers import DynamicCache
import json
import pandas as pd
import re
import signal
import sys
import torch
from copy import deepcopy,copy
import threading
import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from helper.rollout_merge import merge_rollout_outputs
from helper.tree_verification import pack_tree, trace_verified_path
from helper.rollout_history import RolloutHistory

from helper.fast_lk_reflex import (
    FastLKReflex,
    reflex_or_baseline_probabilities,
    topk_compact_candidates,
)
from helper.sampling import (
    build_sampling_probs,
    sample_from_probs,
    sample_target_from_logits,
)


total_target_time=0
total_draft_time=0
total_check_time=0


def speculative_generate_in_prompt_batches(*, prompt_batch_size, input_ids, attention_mask, **kwargs):
    """Generate one real prompt batch in bounded chunks without changing its order.

    All chunks see the same frozen target/draft weights; the caller performs the
    GRPO update only after the combined rollout has been returned. This lets a
    larger next-rollout batches use bounded decoder invocations. Each response
    belongs to exactly one invocation, including its trajectory-local Reflex A.
    """
    prompt_count = int(input_ids.shape[0])
    if prompt_batch_size <= 0 or prompt_count <= prompt_batch_size:
        return speculative_generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    parts = []
    for start in range(0, prompt_count, prompt_batch_size):
        stop = min(start + prompt_batch_size, prompt_count)
        chunk_mask = attention_mask[start:stop]
        active_columns = torch.nonzero(chunk_mask.any(dim=0), as_tuple=False).flatten()
        if active_columns.numel() == 0:
            raise ValueError('decoder prompt chunk contains no non-padding tokens')
        first_token_column = int(active_columns[0].item())
        part = speculative_generate(
            input_ids=input_ids[start:stop, first_token_column:],
            attention_mask=chunk_mask[:, first_token_column:],
            **kwargs,
        )
        parts.append(part)
    combined = merge_rollout_outputs(parts)
    expected_responses = prompt_count * int(kwargs.get('repeated_generate_nums') or 1)
    if len(combined['generated_token_ids']) != expected_responses:
        raise RuntimeError('chunked speculative rollout changed response count/order')
    return combined


def _cache_num_layers(cache):
    if cache is None:
        return 0
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    if hasattr(cache, "layers"):
        return len(cache.layers)
    return len(cache)


def _cache_get_layer(cache, layer_idx):
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        for key_name, value_name in (
            ("keys", "values"),
            ("key_cache", "value_cache"),
            ("key_states", "value_states"),
            ("_keys", "_values"),
        ):
            if hasattr(layer, key_name) and hasattr(layer, value_name):
                return getattr(layer, key_name), getattr(layer, value_name)
        if isinstance(layer, (tuple, list)) and len(layer) >= 2:
            return layer[0], layer[1]
    return cache[layer_idx]


def _cache_set_layer(cache, layer_idx, key, value):
    if hasattr(cache, "key_cache"):
        cache.key_cache[layer_idx] = key
        cache.value_cache[layer_idx] = value
        return
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        for key_name, value_name in (
            ("keys", "values"),
            ("key_cache", "value_cache"),
            ("key_states", "value_states"),
            ("_keys", "_values"),
        ):
            if hasattr(layer, key_name) and hasattr(layer, value_name):
                try:
                    setattr(layer, key_name, key)
                    setattr(layer, value_name, value)
                    return
                except (AttributeError, RuntimeError):
                    pass
        if isinstance(layer, list) and len(layer) >= 2:
            layer[0] = key
            layer[1] = value
            return
    try:
        cache[layer_idx] = (key, value)
        return
    except TypeError as exc:
        raise AttributeError(
            "Unsupported transformers cache layout: cannot set layer key/value tensors"
        ) from exc


def _cache_seq_length(cache):
    if cache is None:
        return 0
    if hasattr(cache, "get_seq_length"):
        return cache.get_seq_length()
    if _cache_num_layers(cache) == 0:
        return 0
    key, _ = _cache_get_layer(cache, 0)
    return int(key.shape[-2])


def sampling(
    logits, 
    top_k=None, 
    top_p=None, 
    temperature=0.6, 
    eos_token_id=2
):
    """
    Perform combined top-k and top-p (nucleus) sampling on logits.
    
    Args:
        logits (torch.Tensor): Logits from the model output (shape: [batch_size, seq_len, vocab_size]).
        top_k (int or None): Number of highest probability tokens to consider for top-k sampling.
        top_p (float or None): Cumulative probability threshold for top-p sampling.
        temperature (float): Temperature to adjust the sharpness of the distribution.
        eos_token_id (int): The ID of the end-of-sequence token (used as fallback when logits are invalid).
    
    Returns:
        torch.Tensor: Sampled token indices (shape: [batch_size, seq_len]).
    """
    probs = build_sampling_probs(logits, temperature, top_p, top_k, eos_token_id)
    return sample_from_probs(probs)



def get_adaptive_hyperparameters(bsz, verification_capacity,
                                max_draft_token_length, max_draft_k, max_verification_num,
                                min_draft_token_length, draft_token_length_c):
    if bsz <= 0:
        return 1, 1, 1

    verification_num = max(2, min(math.floor(verification_capacity / bsz), max_verification_num))

    draft_token_length = min(math.floor(math.log2(verification_num/draft_token_length_c)), max_draft_token_length)
    draft_token_length = max(draft_token_length, min_draft_token_length)
    
    draft_k = max(1, min(verification_num-1, max_draft_k))

    tree_node_capacity = draft_k + draft_k * draft_k * max(draft_token_length - 1, 0)
    draft_total_token = min(verification_num - 1, tree_node_capacity)
    
    
    return draft_token_length, draft_k, draft_total_token
    

def speculative_generate(model, input_ids, attention_mask, tokenizer,
                        do_sample=False, repeated_generate_nums=None,
                        temperature=0.8, top_p=0.9, top_k=None, 
                        verification_capacity=160, 
                        max_draft_token_length=5, max_draft_k=8, max_verification_num=160,
                        min_draft_token_length=3, draft_token_length_c=0.75,
                        statistical_time=False,return_all_draft_input=False,
                        max_length=2048,
                        reflex_mode="off", reflex_feature_dim=8,
                        reflex_lr=0.05, reflex_weight_decay=0.0,
                        reflex_seed=42, reflex_profile=False,
                        reflex_diagnostics=False,
                        reflex_backend="triton",
                        reflex_feedback_scope="root",
                        reflex_proposal_strategy="fused",
                        reflex_correction_strategy="serial",
                        reflex_feedback_strategy="serial",
                        reflex_feature_strategy="auto",
                        reflex_update_stream=True,
                        reflex_time_updates=False,
                        kv_gather_strategy="stacked",
                        ):

    reflex_mode = str(reflex_mode).strip().lower()
    if reflex_mode not in {"off", "active"}:
        raise ValueError("reflex_mode must be 'off' or 'active'")
    if reflex_mode == "active" and not getattr(model, "is_eagle3_specforge", False):
        raise ValueError("Fast LK Reflex requires the SpecForge EAGLE-3 draft backend")
    reflex = None
    if reflex_feedback_scope not in {"root", "visited_path"}:
        raise ValueError("reflex_feedback_scope must be root or visited_path")
    if kv_gather_strategy not in {"stacked", "per_layer"}:
        raise ValueError("kv_gather_strategy must be stacked or per_layer")
    compact_to_target = None


    class Node:
        
        def __init__(self,depth,input_id):
            
            self.depth=depth
            self.input_id=input_id
            
            

    def draft_generate(model,next_feature_states,draft_hidden_states,draft_past_key_values_tree,
                        draft_token_length,past_position_ids_tensor,padding_positions,
                        draft_k=4,draft_total_token=32):
        
        global total_check_time
        
        dtype=model.dtype
        device=model.device
        bsz=draft_hidden_states.shape[0]
        
        node_nums=draft_k+draft_k*draft_k*(draft_token_length-1)
        trees=[[0] * node_nums for _ in range(bsz)] if reflex is None else []
        if reflex is not None:
            full_parents = [torch.full((bsz, draft_k), -1, device=device, dtype=torch.long)]
            full_contexts = torch.full((bsz, node_nums), -1, device=device, dtype=torch.long)
            beam_node_ids = torch.arange(draft_k, device=device).expand(bsz, -1)
        parents_list=[]
        
        total_input_ids=[]
        total_position_ids=[]
        confidences=[]
        
        draft_position_ids=past_position_ids_tensor.unsqueeze(-1).repeat(1, draft_k) # (bsz, draft_k)
        total_position_ids.append(draft_position_ids)
        
        if getattr(model, "is_eagle3_specforge", False):
            compact_logits = model.compute_compact_logits(draft_hidden_states)
            if reflex is not None:
                next_token_values, _, draft_next_token = reflex.propose(
                    compact_logits, draft_hidden_states, draft_k, compact_to_target, root=True)
            else:
                draft_logits = reflex_or_baseline_probabilities(compact_logits)
                next_token_values, _, draft_next_token = topk_compact_candidates(
                    draft_logits, compact_to_target, draft_k)
        else:
            draft_logits = model.lm_head(draft_hidden_states.to(model.target_model.dtype))
            draft_logits = draft_logits.softmax(dim=-1)
            next_token_values, draft_next_token = torch.topk(draft_logits, k=draft_k, dim=-1)
        draft_confidences=next_token_values.view(bsz, -1) # (bsz, draft_k)
        
        past_kv_len=draft_past_key_values_tree[0][0].shape[-2]
        init_kv_len=draft_past_key_values_tree[0][0].shape[-2]

        for idx_batch in range(bsz if reflex is None else 0):
            
            for idx_k in range(draft_k):
                node=Node(depth=0, input_id=None)
                trees[idx_batch][idx_k]=node
                
        draft_next_token=draft_next_token.view(bsz, -1) # (bsz, draft_k)
        bsz, _, hidden_size = next_feature_states.shape
        next_feature_states=next_feature_states.expand(bsz, draft_k, hidden_size)

        total_input_ids.append(draft_next_token) # (1, bsz, draft_k)
        confidences.append(draft_confidences) # (1, bsz, draft_k)
        attention_seen_indices=torch.arange(past_kv_len, past_kv_len+draft_k, device=device).unsqueeze(-1).unsqueeze(0).repeat(bsz, 1, 1) # (bsz, draft_k, 1)
            
        for idx_token in range(1,draft_token_length):
            
            draft_position_ids=draft_position_ids+1
            total_position_ids.append(draft_position_ids.repeat(1, draft_k))
            
            min_dtype=torch.finfo(dtype).min
            q_length=draft_k
            
            draft_attention_mask=torch.zeros((q_length, past_kv_len+q_length),dtype=dtype,device=device)
            draft_attention_mask[..., init_kv_len:]=min_dtype
            
            draft_attention_mask=draft_attention_mask.unsqueeze(0).unsqueeze(0).repeat(bsz, 1, 1, 1)

            zeros=torch.zeros((bsz, 1, q_length, idx_token), dtype=dtype, device=device)

            draft_attention_mask.scatter_(dim=-1, index=attention_seen_indices.unsqueeze(1), src=zeros)

            if isinstance(padding_positions, torch.Tensor):
                padding_positions_tensor=padding_positions
            else:
                padding_positions_indices=[]
        
                for batch_id, pad_positions in enumerate(padding_positions):
                    for pos in pad_positions:
                        padding_positions_indices.append([batch_id, pos])

                if padding_positions_indices:  

                    padding_positions_indices=torch.tensor(padding_positions_indices, device=model.device)
                    
                padding_positions_tensor=padding_positions_indices
                
            if isinstance(padding_positions_tensor, torch.Tensor):
                draft_attention_mask[padding_positions_tensor[:, 0], 0, :, padding_positions_tensor[:, 1]] = min_dtype

            if statistical_time:
                torch.cuda.synchronize()
                check_time_start=time.time()
            
            draft_outputs=model(hidden_states=next_feature_states,input_ids=draft_next_token,
                                attention_mask=draft_attention_mask,use_cache=True,
                                past_key_values=draft_past_key_values_tree,position_ids=draft_position_ids)
            
            if statistical_time:
                torch.cuda.synchronize()
                total_check_time+=time.time()-check_time_start
            
            draft_past_key_values_tree=draft_outputs['past_key_values']
            draft_hidden_states=draft_outputs['hidden_states']
            next_feature_states=draft_outputs['next_feature_states'] # (bsz, seq, hidden_size)
            
            if getattr(model, "is_eagle3_specforge", False):
                compact_logits = model.compute_compact_logits(draft_hidden_states)
                if reflex is not None:
                    context_ids = 1 + (idx_token - 1) * draft_k + torch.arange(draft_k, device=device)
                    full_contexts.scatter_(1, beam_node_ids, context_ids.expand(bsz, -1))
                    full_parents.append(beam_node_ids.repeat_interleave(draft_k, dim=1))
                    next_token_values, _, draft_next_token = reflex.propose(
                        compact_logits, draft_hidden_states, draft_k, compact_to_target)
                else:
                    draft_logits = reflex_or_baseline_probabilities(compact_logits)
                    next_token_values, _, draft_next_token = topk_compact_candidates(
                        draft_logits, compact_to_target, draft_k)
            else:
                draft_logits = model.lm_head(draft_hidden_states.to(model.target_model.dtype))
                draft_logits = draft_logits.softmax(dim=-1)
                next_token_values, draft_next_token = torch.topk(draft_logits, k=draft_k, dim=-1)

            draft_confidences=draft_confidences.unsqueeze(-1)*next_token_values
            draft_top_k_token_values, draft_top_k_token_indices=torch.topk(draft_confidences.view(bsz, -1), k=draft_k, dim=-1) # (bsz, draft_k)
            if reflex is not None:
                beam_node_ids = draft_k + draft_k * draft_k * (idx_token - 1) + draft_top_k_token_indices

            past_kv_len=draft_past_key_values_tree[0][0].shape[-2]
            
            for idx_tree in range(len(trees)):
                for idx_seq in range(draft_k):
                    for idx_k in range(draft_k):
                        
                        index=draft_k + draft_k*draft_k*(idx_token-1) + idx_seq*draft_k + idx_k
                        node=Node(depth=idx_token, input_id=None)

                        assert trees[idx_tree][index]==0
                        trees[idx_tree][index]=node
                        
            total_input_ids.append(draft_next_token.view(bsz, -1)) # (draft_token_length, bsz, draft_k or draft_k*draft_k)
            confidences.append(draft_confidences.view(bsz, -1)) # (draft_token_length, bsz, draft_k or draft_k*draft_k)
            
            draft_next_token=draft_next_token.view(bsz, -1).gather(index=draft_top_k_token_indices, dim=-1)
            draft_confidences=draft_confidences.view(bsz, -1).gather(index=draft_top_k_token_indices, dim=-1)
            
            draft_top_k_token_indices_div_k=draft_top_k_token_indices//draft_k
            
            draft_top_k_token_indices_expanded = draft_top_k_token_indices_div_k.unsqueeze(-1).expand(bsz, draft_k, next_feature_states.shape[-1])
            next_feature_states=next_feature_states.gather(index=draft_top_k_token_indices_expanded, dim=-2)
            
            draft_top_k_token_indices_expanded = draft_top_k_token_indices_div_k.unsqueeze(-1).expand(bsz, draft_k, idx_token)
            attention_seen_indices=attention_seen_indices.gather(index=draft_top_k_token_indices_expanded, dim=-2)
            
            cur_attention_seen_indices=torch.arange(past_kv_len, past_kv_len+draft_k, device=device).unsqueeze(-1).unsqueeze(0).repeat(bsz, 1, 1) # (bsz, draft_k, 1)
            attention_seen_indices=torch.concat([attention_seen_indices, cur_attention_seen_indices], dim=-1)
            
            parents_list.append(draft_top_k_token_indices)
            
        if draft_token_length>1 and reflex is None:
            parents_list=torch.stack(parents_list, dim=1) # (bsz, draft_token_length-1, draft_k)    
            parents_list=parents_list.to(torch.int16).cpu().tolist()
                
        total_input_ids=torch.concat(total_input_ids, dim=-1) # (bsz, node_nums)
        total_position_ids=torch.concat(total_position_ids, dim=1) # (bsz, node_nums)
        confidences=torch.concat(confidences, dim=-1) # (bsz, node_nums)
        draft_total_token=min(int(draft_total_token), int(confidences.shape[-1]))
        
        chosen_index=torch.topk(confidences, k=draft_total_token, dim=-1)
        chosen_index, _=torch.sort(chosen_index.indices, dim=-1, descending=False)
        if reflex is not None:
            tensor_tree = pack_tree(torch.cat(full_parents, dim=1), full_contexts, chosen_index,
                                    total_input_ids, draft_token_length)
            return {'trees': [], 'trees_chosen_index': None, 'tensor_tree': tensor_tree,
                    'next_token_trees': total_input_ids.gather(1, chosen_index),
                    'target_position_ids': total_position_ids.gather(1, chosen_index)}
        chosen_index_list=chosen_index.to(torch.int16).cpu().tolist()
        
        for idx_tree, tree in enumerate(trees):
            for index in chosen_index_list[idx_tree]:
                node=tree[index]
                
                node.child=[]
                node.input_id=total_input_ids[idx_tree][index]
                
                if index < draft_k:
                    node.parent=[]
                    
                elif index < (draft_k+draft_k*draft_k):
                    parent_index=(index-draft_k)//draft_k
                    
                    node.parent=[parent_index]
                    tree[parent_index].child.append(index)
                    
                else:
                    
                    parent_index1=(index-draft_k) // (draft_k*draft_k)
                    parent_index2=(index-draft_k - parent_index1*draft_k*draft_k)//draft_k
                    parent_index2=parents_list[idx_tree][parent_index1-1][parent_index2]
                    
                    parent_index=draft_k + (parent_index1-1)*draft_k*draft_k + parent_index2

                    node.parent=tree[parent_index].parent+[parent_index]
                    tree[parent_index].child.append(index)
                    
                    
        next_token_trees=total_input_ids.gather(index=chosen_index, dim=-1)
        target_position_ids=total_position_ids.gather(index=chosen_index, dim=-1)
                
        return {
            'trees':trees,
            'trees_chosen_index':chosen_index_list,
            'next_token_trees':next_token_trees,
            'target_position_ids':target_position_ids
        }


    def model_forward(model,input_ids,attention_mask,past_key_values,position_ids=None):

        if hasattr(model,'base_model'):
            if hasattr(model.base_model,'model'):
                model=model.base_model.model

        past_seen_tokens = _cache_seq_length(past_key_values)
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Use the public HF model forward instead of calling decoder layers by hand.
        # Qwen/Llama RoPE signatures changed across transformers versions; the public
        # path builds cos/sin with the correct head_dim and still accepts 4D tree masks.
        eagle_layers = getattr(model, "_fastgrpo_eagle3_capture_layers", None)
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "cache_position": cache_position,
            "output_attentions": False,
            "output_hidden_states": eagle_layers is not None,
            "return_dict": True,
        }
        try:
            outputs = model.model(**forward_kwargs)
        except TypeError:
            forward_kwargs.pop("cache_position", None)
            outputs = model.model(**forward_kwargs)

        result = {
            'last_hidden_state': outputs.last_hidden_state,
            'target_hidden_state': outputs.last_hidden_state,
            'past_key_values': outputs.past_key_values,
        }
        if eagle_layers is not None:
            # Hugging Face hidden_states[0] is the embedding output, while
            # SpecForge capture layer ids address decoder layers.
            result['last_hidden_state'] = torch.cat(
                [outputs.hidden_states[int(layer_id) + 1] for layer_id in eagle_layers],
                dim=-1,
            )
        return result


    def get_attention_mask(past_seq_len,q_length,dtype,bsz=1,device='cuda',padding_positions=None):

        min_dtype = torch.finfo(dtype).min
        
        kv_length=past_seq_len+q_length
        
        attention_mask=torch.triu(torch.full((q_length,kv_length),fill_value=min_dtype,dtype=dtype,device=device)
                            , diagonal=kv_length-q_length+1)
        attention_mask=attention_mask.unsqueeze(0).unsqueeze(0).repeat(bsz,1,1,1)
        
        if isinstance(padding_positions, torch.Tensor):
            padding_positions_tensor=padding_positions
        
            attention_mask[padding_positions_tensor[:, 0], 0, :, padding_positions_tensor[:, 1]] = min_dtype

        elif padding_positions:
            
            batch_indices = []
            pos_indices = []

            for batch_id, pad_positions in enumerate(padding_positions):
                for pos in pad_positions:
                    batch_indices.append(batch_id)
                    pos_indices.append(pos)

            if batch_indices: 

                attention_mask[batch_indices, 0, :, pos_indices] = min_dtype
            

        return attention_mask


    if statistical_time:
        torch.cuda.synchronize()
    start_time=time.perf_counter()
    target_past_key_values=DynamicCache()
    avg_acc_length=[0,0]
    total_accepted_draft_tokens=0
    total_proposed_draft_tokens=0
    eos_token_id = tokenizer.eos_token_id
    bsz=input_ids.shape[0]
    end_sig=[0]*bsz
    device=model.target_model.device
    
    transfer_stream = torch.cuda.Stream(device) if reflex_mode == "off" else None
    executor = ThreadPoolExecutor(max_workers=64) if reflex_mode == "off" else None
    update_stream = (torch.cuda.Stream(device) if reflex_mode == "active" and reflex_update_stream
                     and torch.device(device).type == "cuda" else None)
    time_reflex_events = reflex_profile or reflex_time_updates
    profile_overlap_events = [] if time_reflex_events and update_stream is not None else None

    def wait_for_reflex_update():
        if profile_overlap_events is not None:
            main_ready = torch.cuda.Event(enable_timing=True)
            main_ready.record(torch.cuda.current_stream(device))
            profile_overlap_events.append((source_ready, update_done, main_ready))
        torch.cuda.current_stream(device).wait_event(update_done)

    prefill_time_start=time.time()
    target_time_start=time.time()
    
    global total_target_time, total_draft_time, total_check_time
    
    total_target_time, total_draft_time, total_check_time = 0, 0, 0

    all_draft_input_states=None
    all_target_hidden_states=None
    all_draft_input_ids=None

    # A single metadata transfer replaces one GPU scalar sync per prompt token.
    attention_mask = attention_mask.cpu() if reflex_mode == "active" else attention_mask
    position_ids=[torch.sum(item) for item in attention_mask]
    past_position_ids=[item.item()-1 for item in position_ids]

    position_ids=[torch.concat(
        [torch.zeros((input_ids.shape[-1]-item), dtype=torch.long),
         torch.arange(0,item, dtype=torch.long)],dim=-1)
                for item in position_ids]
    position_ids=torch.stack(position_ids,dim=0)

    padding_positions=[]
    for example in attention_mask:
        cur_padding_positions=set()

        for idx, cur_attention_mask in enumerate(example):
            if not cur_attention_mask:
                cur_padding_positions.add(idx)

        padding_positions.append(cur_padding_positions)

    input_ids=input_ids.to(device)
    attention_mask=get_attention_mask(0,attention_mask.shape[-1],model.target_model.dtype,
                                        bsz,device=device,padding_positions=padding_positions)
    position_ids=position_ids.to(device)

    with torch.amp.autocast(str(model.target_model.device),
                                dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        
        target_outputs=model_forward(model.target_model,input_ids=input_ids,attention_mask=attention_mask,
                                past_key_values=target_past_key_values,position_ids=position_ids)
        target_past_key_values = target_outputs['past_key_values']

        feature_states=target_outputs['last_hidden_state']
        target_hidden_states=target_outputs['target_hidden_state']
        target_logits=model.target_model.lm_head(target_hidden_states[:,-1:,:])
    

    if statistical_time:
        torch.cuda.synchronize()
        total_target_time+=time.time()-target_time_start
            
    target_next_token, _ = sample_target_from_logits(
        target_logits,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=eos_token_id,
    )
    if reflex_mode == "active":
        del _, target_logits
    
    
    draft_input_ids=torch.concat([input_ids[:,1:],target_next_token],dim=-1)
    # Row p: shifted token x[p+1], unchanged auxiliary feature and final hidden
    # for x[p]. Online supervision shifts ONLY final hidden after verification
    # rows have been concatenated and padding removed (eagle3_supervision.py).
    draft_attention_mask=attention_mask.to(model.dtype)
        
    initial_history = {'generated_ids': target_next_token}
    if return_all_draft_input:
        initial_history.update(features=feature_states, target_hidden=target_hidden_states,
                               input_ids=draft_input_ids)
    history = RolloutHistory(
        initial_history, repeats=max(1, repeated_generate_nums or 1),
        max_length=max_length + max_draft_token_length + 1)
    del initial_history

    if statistical_time:
        torch.cuda.synchronize()
        draft_time_start=time.time()

    with torch.amp.autocast(str(model.target_model.device),
                                dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        
        draft_outputs=model(hidden_states=feature_states.to(model.dtype),input_ids=draft_input_ids,
                            attention_mask=draft_attention_mask,position_ids=position_ids,use_cache=True)
    
    if statistical_time:
        torch.cuda.synchronize()
        total_draft_time+=time.time()-draft_time_start

    draft_past_key_values=draft_outputs['past_key_values']
    draft_hidden_states=draft_outputs['hidden_states'][:,-1:,:]
    next_feature_states=draft_outputs['next_feature_states'][:,-1:,:]
    
    if repeated_generate_nums is not None and repeated_generate_nums>1:
        
        target_next_token=target_next_token.repeat_interleave(repeated_generate_nums,dim=0)
        target_past_key_values.batch_repeat_interleave(repeated_generate_nums)
        
        new_past_key_values=[]
        for cur_past_key_values in draft_past_key_values:
            cur_past_key_values=[cur_past_key_values[0].repeat_interleave(repeated_generate_nums,dim=0),
                                cur_past_key_values[1].repeat_interleave(repeated_generate_nums,dim=0)]
            new_past_key_values.append(cur_past_key_values)
        draft_past_key_values=new_past_key_values
        
        draft_hidden_states=draft_hidden_states.repeat_interleave(repeated_generate_nums,dim=0)
        next_feature_states=next_feature_states.repeat_interleave(repeated_generate_nums,dim=0)

        bsz*=repeated_generate_nums
        end_sig=[0]*bsz
        
        new_past_position_ids=[]
        for cur_past_position_ids in past_position_ids:
            for _ in range(repeated_generate_nums):
                new_past_position_ids.append(cur_past_position_ids)
        past_position_ids=new_past_position_ids
        
        new_padding_positions=[]
        for cur_padding_positions in padding_positions:
            for _ in range(repeated_generate_nums):
                new_padding_positions.append(deepcopy(cur_padding_positions))
        padding_positions=new_padding_positions
        
            
    draft_token_length, draft_k, draft_total_token = get_adaptive_hyperparameters(bsz, verification_capacity,
                                max_draft_token_length, max_draft_k, max_verification_num,
                                min_draft_token_length, draft_token_length_c)
            
    padding_positions_indices=[]
        
    for batch_id, pad_positions in enumerate(padding_positions):
        for pos in pad_positions:
            padding_positions_indices.append([batch_id, pos])

    if padding_positions_indices: 

        padding_positions_indices=torch.tensor(padding_positions_indices, device=model.device)
        
    padding_positions_tensor=padding_positions_indices
    
    past_position_ids_tensor=torch.tensor(past_position_ids, dtype=torch.int16).to(device).long()
    
    draft_input_states_dict={}
    target_hidden_states_dict={}
    draft_input_ids_dict={}
    generated_sequences_dict={}
    padding_positions_dict={}
    residual_index=[_ for _ in range(bsz)]
    response_accepted_length_sum=[0 for _ in range(bsz)]
    response_verification_rounds=[0 for _ in range(bsz)]

    if getattr(model, "is_eagle3_specforge", False):
        compact_to_target = model.compact_to_target_ids(device=draft_hidden_states.device)
    if reflex_mode == "active":
        reflex = FastLKReflex(
            feature_dim=reflex_feature_dim,
            learning_rate=reflex_lr,
            weight_decay=reflex_weight_decay,
            seed=reflex_seed,
            profile=reflex_profile,
            diagnostics=reflex_diagnostics,
            backend=reflex_backend,
            feedback_scope=reflex_feedback_scope,
            proposal_strategy=reflex_proposal_strategy,
            correction_strategy=reflex_correction_strategy,
            feedback_strategy=reflex_feedback_strategy,
            feature_strategy=reflex_feature_strategy,
            time_updates=reflex_time_updates,
        )
        reflex.start(
            bsz,
            model.compact_vocab_size,
            int(draft_hidden_states.shape[-1]),
            draft_hidden_states.device,
            max_contexts=1 + max_draft_k * (max_draft_token_length - 1),
            max_path_length=max_draft_token_length + 1,
            max_proposal_contexts=max_draft_k, max_topk=max_draft_k,
        )
        # One asynchronous initialization check per rollout, not per round.
        torch._assert_async(torch.count_nonzero(reflex.state) == 0,
                            'Reflex A must start at zero for every response/rollout')
        mask_rows_capacity = max(verification_capacity + bsz, 2 * bsz)
        # KV includes verification padding, not just real-token max_length.
        # Bound every allowed round so unequal path lengths cannot exhaust the
        # pool and reintroduce large per-round allocations.
        mask_columns_capacity = (input_ids.shape[-1] + max_length * (max_draft_token_length + 1)
                                 + max_verification_num)
        tree_mask_workspace = torch.empty(
            mask_rows_capacity * mask_columns_capacity,
            device=device, dtype=model.target_model.dtype)
        

    if statistical_time:
        torch.cuda.synchronize()
        draft_time_start=time.time()

    with torch.amp.autocast(str(model.target_model.device),
                        dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
    
        
        outputs=draft_generate(model,next_feature_states,draft_hidden_states,draft_past_key_values,
                            draft_token_length,past_position_ids_tensor,padding_positions_tensor,
                            draft_k=draft_k, draft_total_token=draft_total_token)
        
        draft_trees=outputs['trees']
        trees_chosen_index=outputs['trees_chosen_index']
        next_token_trees=outputs['next_token_trees']
        target_position_ids=outputs['target_position_ids']
        tensor_tree = outputs.get('tensor_tree')

    if statistical_time:
        torch.cuda.synchronize()
        total_draft_time+=time.time()-draft_time_start

    total_prefill_time=time.time()-prefill_time_start

    for token_num in range(1,max_length):
        
        past_kv_len=_cache_seq_length(target_past_key_values)
        kv_length=past_kv_len+draft_total_token+1
        q_length=draft_total_token+1

        target_trees=draft_trees
        
        for idx_tree, tree in enumerate(target_trees):
            unique_index=past_kv_len+1
            
            for index in trees_chosen_index[idx_tree]:
                
                node=tree[index]
                node.target_index=unique_index
                unique_index+=1
                
        next_token_trees=torch.concat([target_next_token, next_token_trees], dim=-1) # (bsz, q_length)
        target_position_ids = target_position_ids+2
        target_position_ids = torch.concat([(past_position_ids_tensor+1).unsqueeze(-1), target_position_ids], dim=-1) # (bsz, q_length)
                        
        min_dtype=torch.finfo(model.target_model.dtype).min
        
        if reflex is not None:
            target_attention_mask = tensor_tree.attention_mask(past_kv_len, model.target_model.dtype,
                kernels=reflex._kernels, workspace=tree_mask_workspace)
        else:
            target_attention_mask=torch.zeros((q_length, kv_length), dtype=model.target_model.dtype,device=device)
            target_attention_mask[..., past_kv_len+1:]=min_dtype
            target_attention_mask=target_attention_mask.unsqueeze(0).unsqueeze(0).repeat(bsz,1,1,1)
        
        indices=[]
        
        for idx_tree, tree in enumerate(target_trees):
            
            for index in trees_chosen_index[idx_tree]:
                
                node=tree[index]
                
                cur_index=node.target_index-past_kv_len
                for index in node.parent:
                    
                    seen_token_index=tree[index].target_index
                    indices.append([idx_tree, cur_index, seen_token_index])

                indices.append([idx_tree, cur_index, node.target_index])
                        
        if indices:
            indices=torch.tensor(indices, dtype=torch.int16).to(device).long()
            target_attention_mask[indices[:, 0], 0, indices[:, 1], indices[:, 2]] = 0
            
        if isinstance(padding_positions_tensor, torch.Tensor):
            target_attention_mask[padding_positions_tensor[:, 0], 0, :, padding_positions_tensor[:, 1]] = min_dtype
            
        def transfer_input_ids(trees, trees_chosen_index, device, stream):
            torch.cuda.set_device(device)
            
            with torch.cuda.stream(stream):
                for idx_tree, tree in enumerate(trees):
                    for index in trees_chosen_index[idx_tree]:
                        
                        node=tree[index]
                        node.input_id=node.input_id.cpu()
                        
            return trees
        
        transfer_thread = (executor.submit(transfer_input_ids, target_trees, trees_chosen_index, device, transfer_stream)
                           if reflex is None else None)

        if statistical_time:
            torch.cuda.synchronize()
            target_time_start=time.time()

        with torch.amp.autocast(str(model.target_model.device),
                            dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
            
            target_outputs=model_forward(model.target_model,input_ids=next_token_trees,attention_mask=target_attention_mask,
                                            past_key_values=target_past_key_values,position_ids=target_position_ids)
            target_past_key_values = target_outputs['past_key_values']
            
            feature_states_tree=target_outputs['last_hidden_state']
            target_hidden_states_tree=target_outputs['target_hidden_state']
            target_outputs_logits=model.target_model.lm_head(target_hidden_states_tree)

            target_next_token_tree, target_sampling_probs = sample_target_from_logits(
                target_outputs_logits,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos_token_id,
                reflex=reflex if reflex_feedback_scope == "root" and update_stream is None else None,
                compact_to_target=compact_to_target,
            )
            if reflex is None:
                _ = target_sampling_probs  # preserve the legacy OFF tensor lifetime
        
        if statistical_time:
            torch.cuda.synchronize()
            total_target_time+=time.time()-target_time_start
        
        if reflex is not None:
            path = trace_verified_path(tensor_tree, target_next_token_tree, eos_token_id,
                                       kernels=reflex._kernels, workspace=reflex.path_workspace)
            feedback_target = target_sampling_probs if do_sample else target_next_token_tree
            if update_stream is not None:
                source_ready = torch.cuda.Event(enable_timing=time_reflex_events)
                source_ready.record(torch.cuda.current_stream(device))
                update_stream.wait_event(source_ready)
                for shared in (feedback_target, compact_to_target, reflex.state,
                               reflex._feedback_raw, reflex._feedback_psi, reflex._feedback_norm,
                               path.packed_indices, path.feedback_contexts):
                    if isinstance(shared, torch.Tensor):
                        shared.record_stream(update_stream)
                if reflex._feedback_workspace is not None:
                    for workspace_tensor in reflex._feedback_workspace:
                        workspace_tensor.record_stream(update_stream)
                with torch.cuda.stream(update_stream):
                    if reflex_feedback_scope == "root":
                        if do_sample:
                            reflex.update_from_target_probs(feedback_target[:, 0], compact_to_target)
                        else:
                            reflex.update_from_target_tokens(feedback_target[:, 0], compact_to_target)
                    else:
                        reflex.update_visited(feedback_target, compact_to_target, path, greedy=not do_sample)
                    update_done = torch.cuda.Event(enable_timing=time_reflex_events)
                    update_done.record(update_stream)
            elif reflex_feedback_scope == "visited_path":
                reflex.update_visited(feedback_target, compact_to_target, path, greedy=not do_sample)
            # The only host packet is small scheduling metadata. Accepted
            # tokens and KV indices stay on GPU through gather/compaction.
            acc_length = path.lengths.cpu().tolist()
            max_acc_length = max(acc_length)
            next_token, chosen_index, newly_padded, last_valid_index = path.padded_gpu(
                past_kv_len, max_acc_length, eos_token_id,
                kernels=reflex._kernels, workspace=reflex.padded_path_workspace)
            scheduling_packet = torch.cat((next_token.gather(1, last_valid_index).eq(eos_token_id).long(),
                                           torch.where(newly_padded, chosen_index, -1)), dim=1).cpu().tolist()
            end_sig = [int(row[0]) for row in scheduling_packet]
            for idx_tree, length in enumerate(acc_length):
                padding_positions[idx_tree].update(index for index in scheduling_packet[idx_tree][1:] if index >= 0)
                total_proposed_draft_tokens += draft_total_token
                total_accepted_draft_tokens += max(length - 1, 0)
                avg_acc_length[0] += length
                avg_acc_length[1] += 1
            # Release full teacher distributions before the next draft, not after it.
            del target_sampling_probs, target_outputs_logits
        else:
            target_trees=transfer_thread.result()
            acc_length=[0]*bsz
            chosen_index=[[] for _ in range(bsz)]
            next_token=[[] for _ in range(bsz)]
            target_next_token_tree_list=target_next_token_tree.tolist()
        
        for idx_tree, tree in enumerate(target_trees):
            
            if end_sig[idx_tree]==0:
                proposed_draft_tokens=len(trees_chosen_index[idx_tree])
                cur_next_token=[target_next_token_tree_list[idx_tree][0]]
                cur_acc_length=1
                cur_chosen_index=[past_kv_len]
                tmp_sig=True
                
                for index in trees_chosen_index[idx_tree]:
                    
                    if index >= draft_k:
                        break
                
                    node=tree[index]
                    
                    if cur_next_token[-1]==node.input_id.item(): 
                        cur_next_token.append(target_next_token_tree_list[idx_tree][node.target_index-past_kv_len])
                        cur_acc_length+=1
                        cur_chosen_index.append(node.target_index)
                        
                        while tmp_sig:
                            tmp_sig=False
                            
                            for child_index in node.child:
                                child=tree[child_index]
                                
                                if cur_next_token[-1]==child.input_id.item(): 
                                    cur_next_token.append(target_next_token_tree_list[idx_tree][child.target_index-past_kv_len])
                                    cur_acc_length+=1
                                    cur_chosen_index.append(child.target_index)
                                    
                                    tmp_sig=True 
                                    node=child
                                    break
                        break

                acc_length[idx_tree]=cur_acc_length
                next_token[idx_tree]=cur_next_token
                chosen_index[idx_tree]=cur_chosen_index
                
                for cur_idx, token in enumerate(next_token[idx_tree]):

                    if token==eos_token_id:
                        end_sig[idx_tree]=1
                        
                        chosen_index[idx_tree]=chosen_index[idx_tree][:cur_idx+1]
                        next_token[idx_tree]=next_token[idx_tree][:cur_idx+1]
                        acc_length[idx_tree]=cur_idx+1
                        break
                
                total_proposed_draft_tokens+=int(proposed_draft_tokens)
                total_accepted_draft_tokens+=max(int(acc_length[idx_tree])-1, 0)
                avg_acc_length[0]=avg_acc_length[0]+acc_length[idx_tree]
                avg_acc_length[1]+=1
                        
            else:
                acc_length[idx_tree]=0

        max_acc_length=max(acc_length)
        for active_index, accepted_length in enumerate(acc_length):
            if accepted_length > 0:
                original_index = residual_index[active_index]
                response_accepted_length_sum[original_index] += int(accepted_length)
                response_verification_rounds[original_index] += 1
        if reflex is None:
            last_valid_index=[max_acc_length-1]*bsz
        for idx_batch in (range(bsz) if reflex is None else ()):

            cur_index_length=len(chosen_index[idx_batch])
            if cur_index_length<max_acc_length:
                padding_num=max_acc_length-cur_index_length
                new_chosen_index=[]
                new_next_token=[]

                cur_index=past_kv_len 
                for index,token in zip(chosen_index[idx_batch],next_token[idx_batch]):
                    if padding_num>0:
                        
                        if index!=cur_index:
                            while cur_index!=index and padding_num!=0:
                                new_chosen_index.append(cur_index)
                                new_next_token.append(eos_token_id)
                                padding_positions[idx_batch].add(cur_index) 
                                cur_index+=1
                                padding_num-=1

                            new_chosen_index.append(index)
                            new_next_token.append(token)
                            last_valid_index[idx_batch]=len(new_chosen_index)-1
                            cur_index+=1
                            
                        else:
                            new_chosen_index.append(index)
                            new_next_token.append(token)
                            last_valid_index[idx_batch]=len(new_chosen_index)-1
                            cur_index+=1

                    else:
                        new_chosen_index.append(index) 
                        new_next_token.append(token)
                        last_valid_index[idx_batch]=len(new_chosen_index)-1
                        
                while padding_num>0:
                    new_chosen_index.append(cur_index)
                    new_next_token.append(eos_token_id)
                    
                    padding_positions[idx_batch].add(cur_index) 
                    cur_index+=1
                    padding_num-=1

                chosen_index[idx_batch]=new_chosen_index
                next_token[idx_batch]=new_next_token

        if reflex is None:
            feature_states_index=[[index-past_kv_len for index in row] for row in chosen_index]
            next_token=torch.tensor(next_token, device=device)
            last_valid_index=torch.tensor(last_valid_index, dtype=torch.int16).unsqueeze(-1).to(device).long()
            feature_states_index=torch.tensor(feature_states_index, dtype=torch.int16).to(device).long()
        else:
            feature_states_index=chosen_index-past_kv_len
        target_next_token=next_token.gather(index=last_valid_index, dim=-1)
        

        
        B,T,D=feature_states_tree.shape
        feature_states_index=feature_states_index.unsqueeze(-1).expand(B,-1,D)
        feature_states=feature_states_tree.gather(dim=1, index=feature_states_index)
        target_hidden_index = feature_states_index[..., :1].expand(
            B, feature_states_index.shape[1], target_hidden_states_tree.shape[-1]
        )
        target_hidden_states = target_hidden_states_tree.gather(
            dim=1, index=target_hidden_index
        )
        
        history_chunk = {'generated_ids': next_token}
        if return_all_draft_input:
            history_chunk.update(features=feature_states, target_hidden=target_hidden_states,
                                 input_ids=next_token)
        history.append(residual_index, history_chunk)
        del history_chunk

        finished_indices = [index for index, finished in enumerate(end_sig) if finished]
        if reflex is not None and finished_indices and update_stream is None:
            # Compact A once for the whole verification round. The remaining
            # FastGRPO tensors are pruned below with their existing semantics.
            reflex.remove_finished(finished_indices)
        
        if 0 not in end_sig:
            if update_stream is not None:
                wait_for_reflex_update()
            break
        real_sequences_length=max(history.lengths['generated_ids'] + input_ids.shape[-1] - len(pad)
                                  for pad in padding_positions)
        
        if real_sequences_length>=max_length:
            if update_stream is not None:
                wait_for_reflex_update()
            break
        
        if finished_indices:
            keep_rows = [row for row, finished in enumerate(end_sig) if not finished]
            keep = torch.tensor(keep_rows, device=device, dtype=torch.long)
            for row in finished_indices:
                original = residual_index[row]
                finished_history = history.finish(original)
                padding_positions_dict[str(original)] = padding_positions[row]
                generated_sequences_dict[str(original)] = finished_history['generated_ids']
                if return_all_draft_input:
                    draft_input_states_dict[str(original)] = finished_history['features']
                    target_hidden_states_dict[str(original)] = finished_history['target_hidden']
                    draft_input_ids_dict[str(original)] = finished_history['input_ids']
                del finished_history
            end_sig = [end_sig[row] for row in keep_rows]
            padding_positions = [padding_positions[row] for row in keep_rows]
            past_position_ids = [past_position_ids[row] for row in keep_rows]
            residual_index = [residual_index[row] for row in keep_rows]
            chosen_index = (chosen_index.index_select(0, keep) if reflex is not None
                            else [chosen_index[row] for row in keep_rows])
            for layer in range(_cache_num_layers(target_past_key_values)):
                key, value = _cache_get_layer(target_past_key_values, layer)
                _cache_set_layer(target_past_key_values, layer,
                                 key.index_select(0, keep), value.index_select(0, keep))
            draft_past_key_values = [[key.index_select(0, keep), value.index_select(0, keep)]
                                     for key, value in draft_past_key_values]
            next_token = next_token.index_select(0, keep)
            target_next_token = target_next_token.index_select(0, keep)
            feature_states = feature_states.index_select(0, keep)
            target_hidden_states = target_hidden_states.index_select(0, keep)
            last_valid_index = last_valid_index.index_select(0, keep)
            bsz = len(keep_rows)
            draft_token_length, draft_k, draft_total_token = get_adaptive_hyperparameters(
                bsz, verification_capacity, max_draft_token_length, max_draft_k,
                max_verification_num, min_draft_token_length, draft_token_length_c)
                
                
                
        padding_positions_indices=[]
        
        for batch_id, pad_positions in enumerate(padding_positions):
            for pos in pad_positions:
                padding_positions_indices.append([batch_id, pos])

        if padding_positions_indices: 

            padding_positions_indices=torch.tensor(padding_positions_indices, device=model.device)
            
        padding_positions_tensor=padding_positions_indices

        if reflex is not None:
            # All accepted indices remain on GPU. Only the common-prefix
            # length is materialized on host for HF DynamicCache.crop().
            positions = torch.arange(max_acc_length, device=device)[None, :] + past_kv_len
            extension = int((chosen_index == positions).long().cumprod(1).sum(1).min().item())
            prefix_length = min(_cache_seq_length(target_past_key_values), past_kv_len + extension)
            full_chosen_length = past_kv_len + max_acc_length
        else:
            for idx_batch in range(bsz):
                chosen_index[idx_batch]=[x for x in range(past_kv_len)]+chosen_index[idx_batch]
            prefix_length=_cache_seq_length(target_past_key_values)
            for idx_batch in range(len(chosen_index)):
                the_prefix_length=0
                for idx,index in enumerate(chosen_index[idx_batch]):
                    if idx==index:
                        the_prefix_length=idx+1
                    else:
                        break
                prefix_length=min(the_prefix_length,prefix_length)
            full_chosen_length = len(chosen_index[0])

        if prefix_length==full_chosen_length:
            target_past_key_values.crop(prefix_length)
            
        else:
            if reflex is not None:
                gather_index = chosen_index[:, prefix_length - past_kv_len:] - prefix_length
            else:
                suffix_indices=[[x-prefix_length for x in item[prefix_length:]] for item in chosen_index]
                gather_index=torch.tensor(suffix_indices,device=device)

            if kv_gather_strategy == "per_layer":
                for layer in range(_cache_num_layers(target_past_key_values)):
                    key, value = _cache_get_layer(target_past_key_values, layer)
                    heads, head_dim = key.shape[1], key.shape[-1]
                    index_expanded = gather_index[:, None, :, None].expand(bsz, heads, -1, head_dim)
                    new_key = torch.cat((key[..., :prefix_length, :],
                                         key[..., prefix_length:, :].gather(-2, index_expanded)), dim=-2)
                    new_value = torch.cat((value[..., :prefix_length, :],
                                           value[..., prefix_length:, :].gather(-2, index_expanded)), dim=-2)
                    _cache_set_layer(target_past_key_values, layer, new_key, new_value)
            else:
                target_past_key_tensor=torch.stack(
                    [_cache_get_layer(target_past_key_values, idx_layer)[0]
                     for idx_layer in range(_cache_num_layers(target_past_key_values))], dim=0)
                target_past_value_tensor=torch.stack(
                    [_cache_get_layer(target_past_key_values, idx_layer)[1]
                     for idx_layer in range(_cache_num_layers(target_past_key_values))], dim=0)
                L, B, H, T, D = target_past_key_tensor.shape
                index_expanded = gather_index.unsqueeze(1).unsqueeze(-1).unsqueeze(0).expand(L, B, H, -1, D)
                prefix_key=target_past_key_tensor[..., :prefix_length,:]
                prefix_value=target_past_value_tensor[..., :prefix_length,:]
                suffix_key=target_past_key_tensor[..., prefix_length:,:].gather(-2, index_expanded)
                suffix_value=target_past_value_tensor[..., prefix_length:,:].gather(-2, index_expanded)
                new_key=torch.concat([prefix_key,suffix_key],dim=-2)
                new_value=torch.concat([prefix_value,suffix_value],dim=-2)
                for layer in range(L):
                    _cache_set_layer(target_past_key_values, layer, new_key[layer], new_value[layer])
                
        draft_attention_mask=get_attention_mask(draft_past_key_values[0][0].shape[-2], max_acc_length,
                                                model.dtype, bsz, padding_positions=padding_positions_tensor)
        draft_position_ids=[[] for _ in range(bsz)]
        
        assert _cache_seq_length(target_past_key_values)==draft_past_key_values[0][0].shape[-2]+max_acc_length
        
        draft_position_ids=[[] for _ in range(bsz)]
        
        for idx_batch in range(bsz):
            cur_position_ids=past_position_ids[idx_batch]
            cur_index=draft_past_key_values[0][0].shape[-2]
            
            for idx_token in range(max_acc_length):
                if cur_index not in padding_positions[idx_batch]:
                    cur_position_ids+=1
                    
                draft_position_ids[idx_batch].append(cur_position_ids)
                cur_index+=1
                
            past_position_ids[idx_batch]=draft_position_ids[idx_batch][-1]
            
        draft_position_ids=torch.tensor(draft_position_ids, dtype=torch.int16).to(device).long()
        past_position_ids_tensor=draft_position_ids[:, -1]
        
        if statistical_time:
            torch.cuda.synchronize()
            draft_time_start=time.time()

        with torch.amp.autocast(str(model.target_model.device),
                            dtype=torch.bfloat16 if model.dtype==torch.bfloat16 else torch.float16):
        
            if statistical_time:
                torch.cuda.synchronize()
                check_time_start=time.time()

            draft_outputs=model(hidden_states=feature_states.to(model.dtype),input_ids=next_token,
                        attention_mask=draft_attention_mask,use_cache=True,
                        position_ids=draft_position_ids,past_key_values=draft_past_key_values)
            
            if statistical_time:
                torch.cuda.synchronize()
                total_check_time+=time.time()-check_time_start 

            draft_past_key_values=draft_outputs['past_key_values']
            
            B, S, D =draft_outputs['hidden_states'].shape
            last_valid_index=last_valid_index.unsqueeze(-1).expand(-1,-1,D) # [bsz, 1] -> [bsz, 1, D]
            draft_hidden_states=draft_outputs['hidden_states'].gather(index=last_valid_index, dim=1)
            next_feature_states=draft_outputs['next_feature_states'].gather(index=last_valid_index, dim=1)

            # Accepted gather, KV pruning, masks and committed-token draft
            # forward are independent of the new A. The event is needed only
            # before batch-compacting A or issuing the next proposal.
            if update_stream is not None:
                wait_for_reflex_update()
                if finished_indices:
                    reflex.remove_finished(finished_indices)

            outputs=draft_generate(model, next_feature_states, draft_hidden_states, draft_past_key_values,
                                draft_token_length, past_position_ids_tensor, padding_positions_tensor,
                                draft_k=draft_k, draft_total_token=draft_total_token)
            
            draft_trees=outputs['trees']
            trees_chosen_index=outputs['trees_chosen_index']
            next_token_trees=outputs['next_token_trees']
            target_position_ids=outputs['target_position_ids']
            tensor_tree = outputs.get('tensor_tree')

        if statistical_time:
            torch.cuda.synchronize()
            total_draft_time+=time.time()-draft_time_start
            
            
    post_time_start=time.time()
            
    for idx_batch in range(bsz):
        delete_idx=idx_batch
        ori_idx=residual_index[idx_batch] 
        
        padding_positions_dict[str(ori_idx)]=padding_positions[delete_idx]
        finished_history = history.finish(ori_idx)
        generated_sequences_dict[str(ori_idx)]=finished_history['generated_ids']
        
        if return_all_draft_input:
            draft_input_states_dict[str(ori_idx)]=finished_history['features']
            target_hidden_states_dict[str(ori_idx)]=finished_history['target_hidden']
            draft_input_ids_dict[str(ori_idx)]=finished_history['input_ids']
        del finished_history
    del history
        
    bsz=len(generated_sequences_dict)

    if return_all_draft_input:

        all_draft_input_states_without_padding=[]
        all_target_hidden_states_without_padding=[]
        all_draft_input_ids_without_padding=[]
        
        for idx_batch in range(bsz):
            chosen_index=[]
            cur_draft_input_states=draft_input_states_dict.pop(str(idx_batch))
            cur_target_hidden_states=target_hidden_states_dict.pop(str(idx_batch))
            cur_draft_input_ids=draft_input_ids_dict.pop(str(idx_batch))
            
            for index in range(cur_draft_input_ids.shape[-1]):
                if index not in padding_positions_dict[str(idx_batch)]:
                    chosen_index.append(index)
                    
            if len(chosen_index) == cur_draft_input_ids.shape[-1]:
                all_draft_input_states_without_padding.append(cur_draft_input_states)
                all_target_hidden_states_without_padding.append(cur_target_hidden_states)
                all_draft_input_ids_without_padding.append(cur_draft_input_ids)
            else:
                all_draft_input_states_without_padding.append(cur_draft_input_states[chosen_index,:])
                all_target_hidden_states_without_padding.append(cur_target_hidden_states[chosen_index,:])
                all_draft_input_ids_without_padding.append(cur_draft_input_ids[chosen_index])
            
        all_draft_input_states=all_draft_input_states_without_padding
        all_target_hidden_states=all_target_hidden_states_without_padding
        all_draft_input_ids=all_draft_input_ids_without_padding
        for row_features, row_teacher, row_tokens in zip(
            all_draft_input_states, all_target_hidden_states, all_draft_input_ids
        ):
            if not (row_features.shape[0] == row_teacher.shape[0] == row_tokens.shape[0]):
                raise RuntimeError('decoder returned unequal EAGLE-3 token/feature/teacher rows')
            
    new_padding_positions=[[] for _ in range(bsz)]
    for idx_batch in range(bsz):
        cur_padding_positions=[]
        
        for index in sorted(padding_positions_dict[str(idx_batch)]):
            if (index+1)>=input_ids.shape[-1]: 
                cur_padding_positions.append(index-input_ids.shape[-1]+1)
        new_padding_positions[idx_batch]=cur_padding_positions

    filtered_generated_token_ids=[]
    max_sequence_length=0

    for idx_batch in range(bsz):
        generated_sequence=generated_sequences_dict.pop(str(idx_batch)).tolist()
        cur_position_ids=new_padding_positions[idx_batch]
        
        sequence_without_padding=[]
        cur_padding_index=0
        
        for idx_token, token in enumerate(generated_sequence):
            
            if cur_padding_index<len(cur_position_ids):
                if idx_token==cur_position_ids[cur_padding_index]:
                    cur_padding_index+=1
                    continue
                else:
                    sequence_without_padding.append(token)
            else:
                sequence_without_padding.append(token)
            
            if token==eos_token_id:
                break
            
        filtered_generated_token_ids.append(sequence_without_padding)
        max_sequence_length=max(max_sequence_length,len(sequence_without_padding))

    draft_acceptance_rate = total_accepted_draft_tokens / max(total_proposed_draft_tokens, 1)
    reflex_stats = reflex.finish() if reflex is not None else None
    if reflex is not None:
        reflex.clear()

    result = {
        'generated_token_ids':filtered_generated_token_ids,
        'max_sequence_length':max_sequence_length,
        'total_acc_length':avg_acc_length[0],
        'total_acc':max_sequence_length/token_num,
        'total_decoded_token_num':avg_acc_length[1],
        'total_accepted_draft_tokens':total_accepted_draft_tokens,
        'total_proposed_draft_tokens':total_proposed_draft_tokens,
        'total_accepted_medusa_tokens':total_accepted_draft_tokens,
        'total_proposed_medusa_tokens':total_proposed_draft_tokens,
        'draft_acceptance_rate':draft_acceptance_rate,
        'medusa_acceptance_rate':draft_acceptance_rate,
        # Output IDs have already crossed to CPU above, completing the main
        # stream as required by the caller. No timing-only synchronization.
        'total_time_cost':time.perf_counter()-start_time,
        'target_time_cost':total_target_time,
        'draft_time_cost':total_draft_time,
        'check_time_cost':total_check_time,
        'prefill_time_cost':total_prefill_time,
        'post_time_cost':time.time()-post_time_start,
        'all_draft_input_states':all_draft_input_states,
        'all_target_hidden_states':all_target_hidden_states,
        'all_draft_input_ids':all_draft_input_ids,
        'response_accepted_length_sum':response_accepted_length_sum,
        'response_verification_rounds':response_verification_rounds,
        'response_generated_tokens':[len(item) for item in filtered_generated_token_ids],
        'reflex_updates': 0 if reflex_stats is None else reflex_stats.updates,
        'reflex_backend': 'off' if reflex is None else reflex.backend,
        'reflex_feedback_scope': 'off' if reflex is None else reflex_feedback_scope,
        'reflex_proposal_strategy': 'off' if reflex is None else reflex.proposal_strategy,
        'reflex_correction_strategy': 'off' if reflex is None else reflex.correction_strategy,
        'reflex_feedback_strategy': 'off' if reflex is None else reflex.feedback_strategy,
        'reflex_feature_strategy': 'off' if reflex is None else reflex.feature_strategy,
        'reflex_update_stream': bool(update_stream is not None),
        'reflex_state_initialized_zero': reflex_mode == 'active',
        'reflex_state_cleared': reflex is None or reflex.state is None,
    }
    if reflex_stats is not None and reflex_time_updates:
        result['reflex_update_gpu_ms'] = reflex_stats.update_gpu_ms
        result['reflex_update_count'] = reflex_stats.updates
        result['reflex_update_gpu_ms_per_update'] = (
            reflex_stats.update_gpu_ms / max(reflex_stats.updates, 1)
            if reflex_stats.update_gpu_ms is not None else None)
        if profile_overlap_events:
            exposed_ms = 0.0
            for source_ready, update_done, main_ready in profile_overlap_events:
                # Events are materialized only AFTER the rollout. Main-ready
                # is recorded immediately before the dependency on the new A.
                update_done.synchronize()
                main_ready.synchronize()
                exposed_ms += max(0.0, source_ready.elapsed_time(update_done)
                                  - source_ready.elapsed_time(main_ready))
            result['reflex_update_exposed_ms'] = exposed_ms
    if reflex_stats is not None and reflex_stats.alpha_sum is not None:
        result['reflex_alpha_sum'] = reflex_stats.alpha_sum
        result['reflex_loss_sum'] = reflex_stats.loss_sum
    if reflex_stats is not None and reflex_profile:
        result['reflex_profile_time_ms'] = reflex_stats.profile_time_ms
        result['reflex_profile_sections_ms'] = reflex_stats.profile_sections_ms
        if profile_overlap_events:
            overlap_ms = 0.0
            update_ms = 0.0
            for source_ready, update_done, main_ready in profile_overlap_events:
                update_done.synchronize()  # opt-in post-rollout profiling only
                main_ready.synchronize()
                elapsed_update = source_ready.elapsed_time(update_done)
                elapsed_main = source_ready.elapsed_time(main_ready)
                update_ms += elapsed_update
                overlap_ms += min(elapsed_update, elapsed_main)
            result['reflex_update_stream_ms'] = update_ms
            result['reflex_estimated_overlap_ms'] = overlap_ms
    return result

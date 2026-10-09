"""Original FastGRPO architecture; OPD adds only A and inference KV storage.

The baseline uses the unmodified DraftModel and DraftAttention. Cache variants
below preserve their attention/MLP arithmetic, replacing only KV concatenation.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import torch
from torch import nn
from helper.modeling_draft import (Model, DraftModel, DraftAttention,
                                    apply_rotary_pos_emb, repeat_kv)
from helper.opd_static_cache import OPDStaticCache
from helper.transformers_compat import prepare_target_decoder_api

@dataclass
class CacheSlot:
    cache: OPDStaticCache
    index: int

class CachedDraftAttention(DraftAttention):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if isinstance(past_key_value, CacheSlot):
            kv_seq_len += past_key_value.cache.get_seq_length(past_key_value.index)
        elif past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
            
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if isinstance(past_key_value, CacheSlot):
            key_states, value_states = past_key_value.cache.update(key_states, value_states, past_key_value.index)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads*self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_value

class CachedDraftModel(DraftModel):
    def forward(
            self,
            hidden_states,
            inputs_embeds,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List] = None,
            use_cache: Optional[bool] = None,
    ):
        
        bsz, seq_len, _ = hidden_states.shape
        seq_length_with_past = seq_len
        past_key_values_length = 0


        if past_key_values is not None:
            past_key_values_length = past_key_values.get_seq_length() if isinstance(past_key_values, OPDStaticCache) else past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
            
        if position_ids is None:
            device = hidden_states.device if hidden_states is not None else hidden_states.device
            position_ids = torch.arange(
                past_key_values_length, seq_len + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_len)
        else:
            position_ids = position_ids.view(-1, seq_len).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (bsz, seq_length_with_past), dtype=torch.bool, device=hidden_states.device
            )
        
        if attention_mask.dim() != 4:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (bsz, seq_len), hidden_states, past_key_values_length
            )

        key_value_cache = []
        
        hidden_states = hidden_states.to(self.dtype)
        inputs_embeds = inputs_embeds.to(self.dtype)

        residual = hidden_states
        hidden_states=self.fs(hidden_states,inputs_embeds)
        hidden_states=hidden_states+residual
            
        for idx, decoder_layer in enumerate(self.layers):

            # past_key_values: List [config.num_hidden_layers,2,past_seq_len,config.hidden_size]
            past_key_value = CacheSlot(past_key_values, idx) if isinstance(past_key_values, OPDStaticCache) else (past_key_values[idx] if past_key_values is not None else None)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                key_value_cache += [layer_outputs[1]]
                
        residual = hidden_states
        hidden_states=self.post_norm(hidden_states)
        
        next_feature_states=self.states_mlp(hidden_states)
        hidden_states=self.logits_mlp(hidden_states)

        next_feature_states=next_feature_states+residual
        hidden_states=hidden_states+residual

        next_feature_states = self.states_last_norm(next_feature_states)
        hidden_states = self.logits_last_norm(hidden_states)

        return {
            'hidden_states':hidden_states,
            'past_key_values':past_key_values if isinstance(past_key_values, OPDStaticCache) else key_value_cache,
            'next_feature_states':next_feature_states
        }

class FastGRPOModel(Model):
    def __init__(self, config, target_model, path=None):
        prepare_target_decoder_api(target_model)
        super().__init__(config, target_model, path=path)

    def enable_opd(self, rank=8):
        from helper.opd_reflex import initialize_projector
        if hasattr(self.draft_model, 'opd_projector'):
            raise ValueError('OPD already initialized')
        device = self.lm_head.weight.device
        self.draft_model.register_parameter('opd_projector', nn.Parameter(
            initialize_projector(self.draft_model.hidden_size, rank, head=self.lm_head.weight).to(device)))
        self.draft_model.register_buffer('opd_projector_grad_sum', torch.zeros_like(self.opd_projector), persistent=False)
        self.draft_model.register_buffer('opd_projector_grad_weight', torch.zeros(1,device=device), persistent=False)
        self.register_buffer('full_vocabulary_ids', torch.arange(self.draft_model.vocab_size,device=device), persistent=False)
        self.draft_model.__class__ = CachedDraftModel
        for layer in self.draft_model.layers:
            layer.self_attn.__class__ = CachedDraftAttention

    @property
    def opd_projector(self): return getattr(self.draft_model, 'opd_projector', None)
    @property
    def opd_projector_grad_sum(self): return self.draft_model.opd_projector_grad_sum
    @property
    def opd_projector_grad_weight(self): return self.draft_model.opd_projector_grad_weight

    def get_opd_projector(self, rank):
        if self.opd_projector.shape[1] != rank: raise ValueError('OPD rank mismatch')
        return self.opd_projector

    @torch.no_grad()
    def apply_opd_projector_gradient(self):
        self.opd_projector.grad = (self.opd_projector_grad_sum / self.opd_projector_grad_weight.clamp_min(1.)).clone()
        self.opd_projector_grad_sum.zero_()
        self.opd_projector_grad_weight.zero_()

    @torch.no_grad()
    def load_opd_projector(self, value):
        if value.shape != self.opd_projector.shape: raise ValueError('OPD projector shape mismatch')
        self.opd_projector.copy_(value)

    def load_model(self, path):
        path=Path(path)
        if path.is_dir(): path=path/'draft.pth'
        state=torch.load(path,map_location='cpu',weights_only=True)
        if 'draft_model' not in state:
            raise ValueError('Expected a FastGRPO draft checkpoint with draft_model; pretrain with train_draft.py')
        self.draft_model.load_state_dict(state['draft_model'])

    def save_model(self, path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'draft_model':self.draft_model.state_dict()},path)

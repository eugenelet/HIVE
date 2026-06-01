import torch
from torch import nn
import torch.nn.attention.flex_attention
import torch.nn.functional as F
from typing import Any, Optional, Union, Callable, Tuple
import warnings
import os
import math
from abc import ABC, abstractmethod
import random

from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward, ALL_ATTENTION_FUNCTIONS, Qwen2Attention, Qwen2RMSNorm, Qwen2RotaryEmbedding, apply_rotary_pos_emb, repeat_kv, rotate_half, logger
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.activations import ACT2FN

from llava_next.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava_next.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava_next.model.multimodal_encoder.clip_encoder import CLIPTextTower
from llava_next.model.multimodal_projector.builder import build_text_projector

from llava_next.utils import rank0_print

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match=None):
    if keys_to_match is None:
        to_return = {k: t for k, t in named_params}
    else:
        to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

class CrossAttentionWithoutValue(nn.Module):
    def __init__(self, model_config):
        super(CrossAttentionWithoutValue, self).__init__()
        # Linear layers for transforming query and key
        self.query_proj = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.key_proj = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        
    def forward(self, query_states, key_states, valid_key_count=None):
        """
        query_states: (bsz, q_len, 1, hidden_size)
        key_states: (bsz, q_len, n_image + 1, hidden_size)
        valid_key_count: (bsz,) number of valid keys for each batch
        """
        # Transform query and key
        query_states = self.query_proj(query_states)  # (bsz, q_len, 1, hidden_size)
        key_states = self.key_proj(key_states)      # (bsz, q_len, n_image + 1, hidden_size)

        # Compute attention scores (dot product between query and key, scaled)
        # We need to transpose key_states appropriately for the dot product
        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1))  # (bsz, q_len, 1, n_image + 1)
        attention_scores = attention_scores / math.sqrt(key_states.size(-1))  # Scaling by sqrt of hidden size

        # Create a mask for valid keys if valid_key_count is provided
        if valid_key_count is not None:
            # Create a mask of shape (bsz, q_len, n_image + 1)
            key_mask = torch.arange(key_states.size(-2), device=key_states.device).unsqueeze(0) < valid_key_count.unsqueeze(1)
            key_mask = key_mask.unsqueeze(1).unsqueeze(2)  # Broadcast to (bsz, q_len, 1, n_image + 1)

            # Apply a large negative value (-inf) to the invalid positions in attention_scores
            attention_scores = attention_scores.masked_fill(~key_mask, float('-inf'))

        # Apply softmax to get the attention weights
        attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        return attention_weights

    # def forward(self, query_states, key_states, valid_key_count=None):
    #     # Transform query and key
    #     query_states = self.input_layernorm(query_states)
    #     key_states = self.input_layernorm(key_states)
    #     query_states = self.query_proj(query_states)  # (batch_size, query_len, embed_dim)
    #     key_states = self.key_proj(key_states)    # (batch_size, key_len, embed_dim)

    #     # Compute attention scores (dot product between query and key, scaled)
    #     attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1))
    #     attention_scores = attention_scores / math.sqrt(key_states.size(-1))

    #     # Create a mask for valid keys if valid_key_count is provided
    #     if valid_key_count is not None:
    #         # Create a mask of shape (batch_size, 1, key_len)
    #         key_mask = torch.arange(key_states.size(1), device=key_states.device).unsqueeze(0) < valid_key_count.unsqueeze(1)
    #         key_mask = key_mask.unsqueeze(1)  # Broadcast to (batch_size, query_len, key_len)

    #         # Apply a large negative value (-inf) to the invalid positions in attention_scores
    #         attention_scores = attention_scores.masked_fill(~key_mask, float('-inf'))

    #     # Apply softmax to get the attention weights
    #     attention_weights = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

    #     return attention_weights

class PromptAwareLocalAttention(torch.nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim):
        super(PromptAwareLocalAttention, self).__init__()
        self.visual_proj = torch.nn.Linear(visual_dim, hidden_dim)
        self.text_proj = torch.nn.Linear(text_dim, hidden_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))

    def forward(self, visual_features, text_features):
        """
        visual_features: Tensor of shape (batch_size, N, input_dim) - visual features (I)
        text_features: Tensor of shape (batch_size, M, input_dim) - text features (Y)
        """
        # Apply linear transformations
        hidden_visual = self.visual_proj(visual_features)  # Shape: (batch_size, N, hidden_dim)
        hidden_text = self.text_proj(text_features)    # Shape: (batch_size, M, hidden_dim)

        # Compute similarity scores (S_ij) using dot product and scale
        similarity_scores = torch.bmm(hidden_text, hidden_visual.transpose(1, 2)) / self.scale  # Shape: (batch_size, M, N)
        batch_size, M, N = similarity_scores.shape
        flattened_scores = similarity_scores.view(batch_size, -1)  # Shape: (batch_size, M * N)

        # Apply softmax to get attention scores (s_ij)
        attention_scores = F.softmax(flattened_scores, dim=-1, dtype=torch.float32).to(hidden_visual.dtype).view(batch_size, M, N)  # Shape: (batch_size, M, N)
        attention_scores = attention_scores.sum(dim=1)  # Shape: (batch_size, N)

        return attention_scores


class Qwen2RotaryEmbedding_3D(Qwen2RotaryEmbedding):
    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension seperately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2

    cos_split = cos.split(mrope_section, dim=-1)
    sin_split = sin.split(mrope_section, dim=-1)

    # cos_forWH = [torch.cat([cos_split[0], cos_split[1]], dim=-1), 
    #              torch.cat([cos_split[2], cos_split[3]], dim=-1)]
    # cos_forWH = [torch.stack([cos_forWH[0][0, :, :, ::2], cos_forWH[0][1, :, :, 1::2]], dim=-2).flatten(-2, -1), 
    #              torch.stack([cos_forWH[1][0, :, :, ::2], cos_forWH[1][1, :, :, 1::2]], dim=-2).flatten(-2, -1)]
    # cos = torch.cat(cos_forWH, dim=-1).unsqueeze(unsqueeze_dim)

    # sin_forWH = [torch.cat([sin_split[0], sin_split[1]], dim=-1), 
    #              torch.cat([sin_split[2], sin_split[3]], dim=-1)]
    # sin_forWH = [torch.stack([sin_forWH[0][0, :, :, ::2], sin_forWH[0][1, :, :, 1::2]], dim=-2).flatten(-2, -1), 
    #              torch.stack([sin_forWH[1][0, :, :, ::2], sin_forWH[1][1, :, :, 1::2]], dim=-2).flatten(-2, -1)]
    # sin = torch.cat(sin_forWH, dim=-1).unsqueeze(unsqueeze_dim)



    cos_forT = [cos_split[2][2], 
                cos_split[5][2]]
    cos_forWH = [torch.cat([cos_split[0], cos_split[1]], dim=-1), 
                 torch.cat([cos_split[3], cos_split[4]], dim=-1)]
    cos_forWH = [torch.stack([cos_forWH[0][0, :, :, ::2], cos_forWH[0][1, :, :, 1::2]], dim=-2).flatten(-2, -1), 
                 torch.stack([cos_forWH[1][0, :, :, ::2], cos_forWH[1][1, :, :, 1::2]], dim=-2).flatten(-2, -1)]
    cos = torch.cat([cos_forWH[0], cos_forT[0], cos_forWH[1], cos_forT[1]], dim=-1).unsqueeze(unsqueeze_dim)

    sin_forT = [sin_split[2][2], 
                sin_split[5][2]]
    sin_forWH = [torch.cat([sin_split[0], sin_split[1]], dim=-1), 
                 torch.cat([sin_split[3], sin_split[4]], dim=-1)]
    sin_forWH = [torch.stack([sin_forWH[0][0, :, :, ::2], sin_forWH[0][1, :, :, 1::2]], dim=-2).flatten(-2, -1), 
                 torch.stack([sin_forWH[1][0, :, :, ::2], sin_forWH[1][1, :, :, 1::2]], dim=-2).flatten(-2, -1)]
    sin = torch.cat([sin_forWH[0], sin_forT[0], sin_forWH[1], sin_forT[1]], dim=-1).unsqueeze(unsqueeze_dim)
    
    # cos = torch.cat([m[i % 2] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
    #     unsqueeze_dim
    # )
    # sin = torch.cat([m[i % 2] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
    #     unsqueeze_dim
    # )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
    

class Qwen2Attention_v2(Qwen2Attention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    
    def __init__(self, cross, config, layer_idx):
        super().__init__(config, layer_idx)
        self.image_embeds = None
        # self.layer_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size))
        # self.layer_feature = nn.Parameter(torch.randn((1, 1, self.config.hidden_size)) * embed_std)
        # self.alpha = None
        # self.layer_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.image_cls_token = None
        self.valid_image_count = None
        self.cross = cross
        
    def set_image_embeds(self, image_embeds, valid_image_count=None):
        if image_embeds is not None:
            # Padding to same number of images
            if valid_image_count is not None:
                image_embeds = torch.split(image_embeds, valid_image_count)
                image_embeds = torch.nn.utils.rnn.pad_sequence(image_embeds, batch_first=True)
                # max_num_images = max(valid_image_count)
                # for i in range(len(image_embeds)):
                #     if len(image_embeds[i]) == 0:
                #         image_embeds[i] = torch.zeros((max_num_images, image_len, self.))
                # image_embeds = torch.stack([torch.cat([
                #     image, 
                #     torch.zeros_like(image[0]).unsqueeze(0).expand(max_num_images - image.shape[0], -1, -1)]
                # , dim=0) for image in image_embeds], dim=0)
                self.valid_image_count = torch.tensor(valid_image_count, dtype=torch.long, device=image_embeds.device)
            self.image_cls_token = image_embeds[:, :, 0, :]  # B, n_image, hidden_size
            self.image_embeds = image_embeds[:, :, 1:, :]   # B, n_image, seq_len, hidden_size
            # self.image_embeds = image_embeds
        else:
            self.image_cls_token = None
            self.image_embeds = None
            self.valid_image_count = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,  # This parameter is unused, can be removed if not needed
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if self.image_embeds is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )
            
        output_attentions = False
        
        bsz, q_len, _ = hidden_states.size()
        _, n_image, image_len, _ = self.image_embeds.size()
        
        # Retrieve CLS tokens
        text_cls_token = hidden_states.unsqueeze(-2)  # (bsz, q_len, 1, hidden_size)
        image_cls_token = self.image_cls_token.unsqueeze(1).expand(-1, q_len, -1, -1)  # (bsz, q_len, n_image, hidden_size)
        cls_token = torch.cat([image_cls_token, text_cls_token], dim=-2)  # (bsz, q_len, n_image+1, hidden_size)
        scores = self.cross(text_cls_token, cls_token, self.valid_image_count+1).squeeze(-2)  # (bsz, q_len, n_image+1)
        
        # Project hidden states once
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings to query_states once
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        
        # Project image embeddings once
        image_key_states = self.k_proj(self.image_embeds)
        image_value_states = self.v_proj(self.image_embeds)
        image_key_states = image_key_states.view(
            bsz, n_image, image_len, self.num_key_value_heads, self.head_dim
        ).transpose(2, 3)  # (bsz, n_image, num_key_value_heads, image_len, head_dim)
        image_value_states = image_value_states.view(
            bsz, n_image, image_len, self.num_key_value_heads, self.head_dim
        ).transpose(2, 3)
        
        # Prepare image_attention_mask once
        if attention_mask is not None:
            image_attention_mask = torch.zeros((bsz, 1, q_len, image_len), dtype=attention_mask.dtype, device=attention_mask.device)
            image_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=-1)
        else:
            image_attention_mask = None
            
        # Initialize attn_output
        attn_output = torch.zeros_like(hidden_states)
        
        for i in range(n_image):
            # Handle past_key_value if present
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                if i == 0:
                    current_key_states, current_value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                else:
                    current_key_states, current_value_states = past_key_value[self.layer_idx]
            else:
                # Clone key_states and value_states to avoid in-place modifications
                current_key_states = key_states.clone()
                current_value_states = value_states.clone()
            
            # Concatenate image key/value states if not the last iteration
            if i != n_image:
                img_key_states = image_key_states[:, i]  # (bsz, num_key_value_heads, image_len, head_dim)
                img_value_states = image_value_states[:, i]
                current_key_states = torch.cat([img_key_states, current_key_states], dim=2)
                current_value_states = torch.cat([img_value_states, current_value_states], dim=2)
                current_attention_mask = image_attention_mask
            else:
                current_attention_mask = attention_mask
            
            # Apply rotary embeddings to key_states
            pos_ids = torch.arange(current_key_states.size(-2), device=position_ids.device).unsqueeze(0)
            cos, sin = self.rotary_emb(current_value_states, pos_ids)
            _, current_key_states = apply_rotary_pos_emb(current_key_states, current_key_states, cos, sin)
            
            # repeat k/v heads if n_kv_heads < n_heads
            current_key_states = repeat_kv(current_key_states, self.num_key_value_groups)
            current_value_states = repeat_kv(current_value_states, self.num_key_value_groups)
            
            # Compute attention output
            attn_weights = torch.matmul(query_states, current_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            if current_attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = current_attention_mask[:, :, :, : current_key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output_result = torch.matmul(attn_weights, current_value_states)

            if attn_output_result.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output_result.size()}"
                )
            
            attn_output_result = attn_output_result.transpose(1, 2).contiguous()
            attn_output_result = attn_output_result.reshape(bsz, q_len, self.hidden_size)
            attn_output_result = self.o_proj(attn_output_result)
            
            # attn_output.append(attn_output_result)
            # Accumulate weighted attention outputs
            weight = scores[:, :, i].unsqueeze(-1)
            attn_output += weight * attn_output_result
            
            # visualize the attention weights
            # if i == 0:
            #     image_attn_weights = attn_weights[0, 0, 1, :image_len].detach().float().cpu().numpy()
            #     dim = int(math.sqrt(image_attn_weights.size))
            #     if dim * dim != image_attn_weights.size:
            #         raise ValueError("The number of image tokens should be a square number")
            #     image_attn_weights = image_attn_weights.reshape(dim, dim)
            #     import matplotlib.pyplot as plt
            #     import matplotlib
            #     matplotlib.use('Agg')
            #     breakpoint()
            #     plt.imshow(image_attn_weights, cmap='viridis')
            #     plt.colorbar()
            #     plt.title(f"layer{self.layer_idx} Attention Weights Visualization")
            #     plt.savefig(f"image/layer{self.layer_idx}_attention_weights.jpg")
            #     plt.close()
                
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

class Qwen2Attention_v3(Qwen2Attention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.image_embeds = None
        # self.layer_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size))
        # self.layer_feature = nn.Parameter(torch.randn((1, 1, self.config.hidden_size)) * embed_std)
        # self.alpha = None
        # self.layer_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.image_cls_token = None
        self.valid_image_count = None
        
    def set_image_embeds(self, image_embeds, valid_image_count=None):
        if image_embeds is not None:
            # Padding to same number of images
            if valid_image_count is not None:
                image_embeds = torch.split(image_embeds, valid_image_count)
                image_embeds = torch.nn.utils.rnn.pad_sequence(image_embeds, batch_first=True)
                # max_num_images = max(valid_image_count)
                # for i in range(len(image_embeds)):
                #     if len(image_embeds[i]) == 0:
                #         image_embeds[i] = torch.zeros((max_num_images, image_len, self.))
                # image_embeds = torch.stack([torch.cat([
                #     image, 
                #     torch.zeros_like(image[0]).unsqueeze(0).expand(max_num_images - image.shape[0], -1, -1)]
                # , dim=0) for image in image_embeds], dim=0)
                self.valid_image_count = torch.tensor(valid_image_count, dtype=torch.long, device=image_embeds.device)
            # self.image_cls_token = image_embeds[:, :, 0, :]  # B, n_image, hidden_size
            # self.image_embeds = image_embeds[:, :, 1:, :]   # B, n_image, seq_len, hidden_size
            self.image_embeds = image_embeds
        else:
            self.image_cls_token = None
            self.image_embeds = None
            self.valid_image_count = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,  # This parameter is unused, can be removed if not needed
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if self.image_embeds is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )
            
        output_attentions = False
        
        bsz, q_len, _ = hidden_states.size()
        _, n_image, image_len, _ = self.image_embeds.size()
        
        # Project hidden states once
        query_states = self.q_proj(hidden_states)
        
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        
        # Apply rotary embeddings to query_states once
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        
        # Project image embeddings once
        image_key_states = self.k_proj(self.image_embeds)
        image_value_states = self.v_proj(self.image_embeds)
        image_key_states = image_key_states.view(
            bsz, n_image, image_len, self.num_key_value_heads, self.head_dim
        ).transpose(2, 3)  # (bsz, n_image, num_key_value_heads, image_len, head_dim)
        image_value_states = image_value_states.view(
            bsz, n_image, image_len, self.num_key_value_heads, self.head_dim
        ).transpose(2, 3)
            
        # Initialize attn_output
        attn_output = torch.zeros_like(hidden_states)
        
        # Concatenate image key/value states if not the last iteration
        img_key_states = image_key_states[:, 0]  # (bsz, num_key_value_heads, image_len, head_dim)
        img_value_states = image_value_states[:, 0]
        
        # Apply rotary embeddings to key_states
        pos_ids = torch.arange(img_key_states.size(-2), device=position_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(img_value_states, pos_ids)
        _, img_key_states = apply_rotary_pos_emb(img_key_states, img_key_states, cos, sin)
        
        # repeat k/v heads if n_kv_heads < n_heads
        img_key_states = repeat_kv(img_key_states, self.num_key_value_groups)
        img_value_states = repeat_kv(img_value_states, self.num_key_value_groups)
        
        # Compute attention output
        attn_weights = torch.matmul(query_states, img_key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output_result = torch.matmul(attn_weights, img_value_states)

        if attn_output_result.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output_result.size()}"
            )
        
        attn_output_result = attn_output_result.transpose(1, 2).contiguous()
        attn_output_result = attn_output_result.reshape(bsz, q_len, self.hidden_size)
        attn_output_result = self.o_proj(attn_output_result)
        
        # visualize the attention weights
        # image_attn_weights = attn_weights[0, 0, 1, :image_len].detach().float().cpu().numpy()
        # dim = int(math.sqrt(image_attn_weights.size))
        # if dim * dim != image_attn_weights.size:
        #     raise ValueError("The number of image tokens should be a square number")
        # image_attn_weights = image_attn_weights.reshape(dim, dim)
        # import matplotlib.pyplot as plt
        # import matplotlib
        # matplotlib.use('Agg')
        # breakpoint()
        # plt.imshow(image_attn_weights, cmap='viridis')
        # plt.colorbar()
        # plt.title(f"layer{self.layer_idx} Attention Weights Visualization")
        # plt.savefig(f"image/layer{self.layer_idx}_attention_weights.jpg")
        # plt.close()
                
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value

class Qwen2FlashAttention2_v1(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_embeds = None
        # self.image_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # self.image_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # std = self.config.initializer_range
        # self.image_k_proj.weight.data.normal_(mean=0.0, std=std)
        # self.image_v_proj.weight.data.normal_(mean=0.0, std=std)
        # self.image_k_proj.bias.data.zero_()
        # self.image_v_proj.bias.data.zero_()
        
    def set_image_embeds(self, image_embeds, layernorm=None):
        if image_embeds is not None:
            self.image_embeds = layernorm(image_embeds)
        else:
            self.image_embeds = None
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        output_attentions = False
        
        if self.image_embeds is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )

        bsz, q_len, _ = hidden_states.size()
        _, image_len, _ = self.image_embeds.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        image_key_states = self.k_proj(self.image_embeds)
        image_value_states = self.v_proj(self.image_embeds)
        image_key_states = image_key_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (bsz, num_key_value_heads, image_len, head_dim)
        image_value_states = image_value_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        position_ids = position_ids + image_len
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        key_states = torch.cat([image_key_states, key_states], dim=2)
        value_states = torch.cat([image_value_states, value_states], dim=2)
        if attention_mask is not None:
            image_attention_mask = torch.ones((bsz, image_len), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([image_attention_mask, attention_mask], dim=-1)
        
        pos_ids = torch.arange(key_states.size(-2), device=position_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(value_states, pos_ids)
        _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)
            
        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=pos_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2SdpaAttention_v1(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_embeds = None
        self.media_offsets = None
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
        self.proj = nn.Sequential(
            nn.Linear(self.config.mm_hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.rotary_emb_2D = Qwen2RotaryEmbedding_3D(config=self.config)
        if getattr(self.config, "mm_kvproj", False):
            self.image_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.image_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # std = self.config.initializer_range
        # self.image_k_proj.weight.data.normal_(mean=0.0, std=std)
        # self.image_v_proj.weight.data.normal_(mean=0.0, std=std)
        # self.image_k_proj.bias.data.zero_()
        # self.image_v_proj.bias.data.zero_()

        
    def set_image_embeds(self, image_embeds, media_offsets=None, layernorm=None):
        if image_embeds is not None:
            self.image_embeds = layernorm(self.proj(image_embeds))
            self.media_offsets = media_offsets
        else:
            self.image_embeds = None
            self.media_offsets = None
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        
        if self.image_embeds is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )

        bsz, q_len, _ = hidden_states.size()
        _, image_len, _ = self.image_embeds.size()
        image_side_len = int(math.sqrt(image_len))

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Shift the token after image to right
        position_ids = position_ids.expand(bsz, -1)
        left_pos_mask = position_ids < self.media_offsets.unsqueeze(1)
        right_pos_mask = ~left_pos_mask
        position_ids = position_ids + right_pos_mask * image_side_len

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        query_states = self.q_norm(query_states)
        
        if q_len > 1:  # Need image embeddings
            if getattr(self.config, "mm_kvproj", False):
                image_key_states = self.image_k_proj(self.image_embeds)
                image_value_states = self.image_v_proj(self.image_embeds)
            else:
                image_key_states = self.k_proj(self.image_embeds)
                image_value_states = self.v_proj(self.image_embeds)
            image_key_states = image_key_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (bsz, num_key_value_heads, image_len, head_dim)
            image_value_states = image_value_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            new_key_states = torch.zeros(
                (bsz, self.num_key_value_heads, q_len + image_len, self.head_dim),
                device=key_states.device,
                dtype=key_states.dtype,
            )
            new_value_states = torch.zeros_like(new_key_states)
            new_position_ids = torch.zeros((3, bsz, q_len + image_len), device=position_ids.device, dtype=position_ids.dtype)
            # new_position_ids = torch.zeros((2, bsz, q_len + image_len), device=position_ids.device, dtype=position_ids.dtype)
            
            for i in range(bsz):
                left_mask = left_pos_mask[i]
                right_mask = right_pos_mask[i]
                media_offset = self.media_offsets[i]

                new_key_states[i] = torch.cat([key_states[i, :, left_mask], image_key_states[i], key_states[i, :, right_mask]], dim=1)
                new_value_states[i] = torch.cat([value_states[i, :, left_mask], image_value_states[i], value_states[i, :, right_mask]], dim=1)

                image_position_id_row_0 = torch.arange(media_offset, media_offset + image_side_len, device=position_ids.device, dtype=position_ids.dtype).repeat_interleave(image_side_len)
                image_position_id_row_1 = torch.arange(media_offset, media_offset + image_side_len, device=position_ids.device, dtype=position_ids.dtype).tile(image_side_len)
                image_position_id_row_2 = torch.full((image_len,), media_offset + image_side_len//2, device=position_ids.device, dtype=position_ids.dtype)
                
                new_position_ids[0, i] = torch.cat([position_ids[i, left_mask], image_position_id_row_0, position_ids[i, right_mask]], dim=0)
                new_position_ids[1, i] = torch.cat([position_ids[i, left_mask], image_position_id_row_1, position_ids[i, right_mask]], dim=0)
                new_position_ids[2, i] = torch.cat([position_ids[i, left_mask], image_position_id_row_2, position_ids[i, right_mask]], dim=0)
                
            key_states = new_key_states
            value_states = new_value_states
            position_ids = new_position_ids

        if position_ids.ndim == 2:
            cos, sin = self.rotary_emb(value_states, position_ids)
            _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        elif position_ids.ndim == 3:
            cos, sin = self.rotary_emb_2D(value_states, position_ids)
            _, key_states = apply_multimodal_rotary_pos_emb(key_states, key_states, cos, sin, [3*self.head_dim//16, 3*self.head_dim//16, 2*self.head_dim//16])
            # _, key_states = apply_multimodal_rotary_pos_emb(key_states, key_states, cos, sin, [self.head_dim//4, self.head_dim//4])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        key_states = self.k_norm(key_states)
        
        if attention_mask is None and q_len != key_states.size(-2) and q_len > 1:
            attention_mask = torch.triu(torch.ones((bsz, 1, q_len, q_len), dtype=key_states.dtype, device=key_states.device), diagonal=1)
            attention_mask = attention_mask.masked_fill(attention_mask == 1, torch.finfo(attention_mask.dtype).min)

        if attention_mask is not None:
            new_attention_mask = torch.zeros(
                (bsz, 1, q_len, attention_mask.size(-1) + image_len),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            for i in range(bsz):
                left_mask = left_pos_mask[i]
                right_mask = right_pos_mask[i]
                media_offset = self.media_offsets[i]
                attention_mask_i = attention_mask[i, 0]

                if torch.any(left_mask):
                    new_attention_mask[i, :, left_mask] = torch.cat([
                        attention_mask_i[left_mask, :media_offset], 
                        torch.full((left_mask.sum(), image_len), torch.finfo(attention_mask.dtype).min, device=attention_mask.device, dtype=attention_mask.dtype),
                        attention_mask_i[left_mask, media_offset:]
                    ], dim=-1)

                if torch.any(right_mask):
                    new_attention_mask[i, :, right_mask] = torch.cat([
                        attention_mask_i[right_mask, :media_offset], 
                        torch.zeros((right_mask.sum(), image_len), device=attention_mask.device, dtype=attention_mask.dtype),
                        attention_mask_i[right_mask, media_offset:]
                    ], dim=-1)
            attention_mask = new_attention_mask

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        # with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION, torch.nn.attention.SDPBackend.CUDNN_ATTENTION]):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    

image_attention_mask = None

class Qwen2Attention_rework(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_embeds = None
        self.media_offsets = None
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
        self.mm_proj = nn.Sequential(
            nn.Linear(self.config.mm_hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )
        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)
        self.rotary_emb_2D = Qwen2RotaryEmbedding_3D(config=self.config)
        if getattr(self.config, "mm_kvproj", False):
            self.image_k_proj = nn.Linear(self.config.hidden_size, self.config.num_key_value_heads * self.head_dim, bias=True)
            self.image_v_proj = nn.Linear(self.config.hidden_size, self.config.num_key_value_heads * self.head_dim, bias=True)

        
    def set_image_embeds(self, image_embeds, media_offsets=None, layernorm=None):
        if image_embeds is not None:
            self.image_embeds = layernorm(self.mm_proj(image_embeds))
            self.media_offsets = media_offsets
        else:
            self.image_embeds = None
            self.media_offsets = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if self.image_embeds is None:
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs
            )
        
        bsz, q_len, _ = hidden_states.size()
        hidden_shape = (bsz, q_len, -1, self.head_dim)
        _, image_len, _ = self.image_embeds.size()
        image_hidden_shape = (bsz, image_len, -1, self.head_dim)
        image_side_len = math.isqrt(image_len)
        # image_side_len = math.isqrt(image_len - (getattr(self.config, "mm_vision_select_feature", "patch") == "cls_patch"))

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        
        position_ids = kwargs.get("position_ids", None)

        # Shift the token after image to right
        position_ids = position_ids.expand(bsz, -1)
        left_pos_mask = position_ids < self.media_offsets.unsqueeze(1)
        right_pos_mask = ~left_pos_mask
        text_indices = position_ids + right_pos_mask * image_len
        image_indices = torch.arange(image_len, dtype=text_indices.dtype, device=text_indices.device).unsqueeze(0).expand(bsz, -1) + self.media_offsets.unsqueeze(1)
        position_ids = position_ids + right_pos_mask * image_side_len
        # position_ids = position_ids + right_pos_mask * \
        #     (image_side_len + (getattr(self.config, "mm_vision_select_feature", "patch") == "cls_patch"))

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        query_states = self.q_norm(query_states)
        
        if q_len > 1:  # Need image embeddings
            if getattr(self.config, "mm_kvproj", False):
                image_key_states = self.image_k_proj(self.image_embeds).view(image_hidden_shape).transpose(1, 2)
                image_value_states = self.image_v_proj(self.image_embeds).view(image_hidden_shape).transpose(1, 2)
            else:
                image_key_states = self.k_proj(self.image_embeds).view(image_hidden_shape).transpose(1, 2)
                image_value_states = self.v_proj(self.image_embeds).view(image_hidden_shape).transpose(1, 2)

            # if getattr(self.config, "mm_vision_select_feature", "patch") == "cls_patch":
            #     image_position_id = torch.stack([
            #         torch.arange(0, image_side_len, device=position_ids.device, dtype=position_ids.dtype).repeat_interleave(image_side_len), 
            #         torch.arange(0, image_side_len, device=position_ids.device, dtype=position_ids.dtype).tile(image_side_len), 
            #         torch.full((image_len - 1,), image_side_len//2, device=position_ids.device, dtype=position_ids.dtype)
            #     ], dim=0)
            #     cls_image_position_id = torch.tensor([image_side_len, image_side_len, image_side_len//2], device=position_ids.device, dtype=position_ids.dtype).unsqueeze(-1)
            #     image_position_id = torch.cat([cls_image_position_id, image_position_id], dim=-1)
            # else: # "patch"
            image_position_id = torch.stack([
                torch.arange(0, image_side_len, device=position_ids.device, dtype=position_ids.dtype).repeat_interleave(image_side_len), 
                torch.arange(0, image_side_len, device=position_ids.device, dtype=position_ids.dtype).tile(image_side_len), 
                torch.full((image_len,), image_side_len//2, device=position_ids.device, dtype=position_ids.dtype)
            ], dim=0)


            new_key_states = torch.empty(
                (bsz, self.config.num_key_value_heads, q_len + image_len, self.head_dim),
                device=key_states.device,
                dtype=key_states.dtype,
            )
            new_value_states = torch.empty_like(new_key_states)
            new_position_ids = torch.empty((3, bsz, q_len + image_len), device=position_ids.device, dtype=position_ids.dtype)

            # Precompute expanded indices for efficiency
            text_indices_expanded = text_indices[:, None, :, None].expand(bsz, self.config.num_key_value_heads, -1, self.head_dim)
            image_indices_expanded = image_indices[:, None, :, None].expand(bsz, self.config.num_key_value_heads, -1, self.head_dim)

            # Perform scatter_ in a more optimized way
            new_key_states.scatter_(2, text_indices_expanded, key_states)
            new_key_states.scatter_(2, image_indices_expanded, image_key_states)

            new_value_states.scatter_(2, text_indices_expanded, value_states)
            new_value_states.scatter_(2, image_indices_expanded, image_value_states)

            # Precompute position IDs expansion (3D indices)
            text_position_ids_expanded = position_ids.expand(3, bsz, -1)
            image_position_id_expanded = image_position_id[:, None].expand(3, bsz, -1) + self.media_offsets[:, None]

            # Scatter text and image positions into the new tensor
            new_position_ids.scatter_(2, text_indices.expand(3, bsz, -1), text_position_ids_expanded)
            new_position_ids.scatter_(2, image_indices.expand(3, bsz, -1), image_position_id_expanded)

            key_states = new_key_states
            value_states = new_value_states
            position_ids = new_position_ids
            
            # for i in range(bsz):
            #     offset = self.media_offsets[i]

            #     new_key_states.append(torch.cat([
            #         key_states[i, :, :offset], 
            #         image_key_states[i], 
            #         key_states[i, :, offset:]
            #     ], dim=1))
            #     new_value_states.append(torch.cat([
            #         value_states[i, :, :offset], 
            #         image_value_states[i], 
            #         value_states[i, :, offset:]
            #     ], dim=1))
            #     new_position_ids.append(torch.cat([
            #         position_ids[i, :offset].expand(image_position_id.size(0), -1), 
            #         image_position_id + offset, 
            #         position_ids[i, offset:].expand(image_position_id.size(0), -1)
            #     ], dim=-1))
                
            # key_states = torch.stack(new_key_states, dim=0)
            # value_states = torch.stack(new_value_states, dim=0)
            # position_ids = torch.stack(new_position_ids, dim=1)

        if position_ids.ndim == 2:
            cos, sin = self.rotary_emb(value_states, position_ids)
            _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
        elif position_ids.ndim == 3:
            cos, sin = self.rotary_emb_2D(value_states, position_ids)
            _, key_states = apply_multimodal_rotary_pos_emb(key_states, key_states, cos, sin, [3*self.head_dim//16, 3*self.head_dim//16, 2*self.head_dim//16])
            # _, key_states = apply_multimodal_rotary_pos_emb(key_states, key_states, cos, sin, [self.head_dim//4, self.head_dim//4])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = self.k_norm(key_states)

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        if attention_mask is None and q_len != key_states.size(-2) and q_len > 1:
            attention_mask = torch.triu(torch.ones((bsz, 1, q_len, q_len), dtype=key_states.dtype, device=key_states.device), diagonal=1)
            attention_mask = attention_mask.masked_fill(attention_mask == 1, torch.finfo(attention_mask.dtype).min)

        if attention_mask is not None:
            global image_attention_mask

            if image_attention_mask is None:
                new_attention_mask = torch.empty(
                    (bsz, 1, q_len, q_len + image_len),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                
                text_indices_expanded = text_indices[:, None, None, :].expand(bsz, 1, q_len, q_len)
                new_attention_mask.scatter_(-1, text_indices_expanded, attention_mask)

                # Create column masks for masked_fill_
                col_idx = torch.arange(q_len + image_len, device=attention_mask.device).expand(bsz, -1)

                # Mask for the image block region (columns ∈ [offset, offset + image_len))
                image_mask = (col_idx >= self.media_offsets.unsqueeze(1)) & (col_idx < self.media_offsets.unsqueeze(1) + image_len)

                # Apply the -inf fill where both conditions hold
                new_attention_mask[:, 0].masked_fill_(image_mask.unsqueeze(1) & left_pos_mask.unsqueeze(-1), torch.finfo(attention_mask.dtype).min)

                # Apply the 0 fill where only the column condition holds
                new_attention_mask[:, 0].masked_fill_(image_mask.unsqueeze(1) & right_pos_mask.unsqueeze(-1), 0.0)

                # for i in range(bsz):
                #     offset = self.media_offsets[i]
                #     attention_mask_i = attention_mask[i, 0]
                #     new_attention_mask_i = new_attention_mask[i, 0]
                    
                #     new_attention_mask_i[:, :offset] = attention_mask_i[:, :offset]
                #     new_attention_mask_i[:offset, offset : offset + image_len] = torch.finfo(attention_mask.dtype).min
                #     new_attention_mask_i[offset:, offset : offset + image_len] = 0.0
                #     new_attention_mask_i[:, offset + image_len:] = attention_mask_i[:, offset:]

                # Cache the computed attention mask.
                image_attention_mask = new_attention_mask

            # Always assign the (possibly cached) image_attention_mask back to attention_mask.
            attention_mask = image_attention_mask

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class Qwen2SdpaAttention_v2(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_embeds = None
        self.media_offsets = None
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=self.config.rms_norm_eps)
        self.image_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.image_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        std = self.config.initializer_range
        self.image_k_proj.weight.data.normal_(mean=0.0, std=std)
        self.image_v_proj.weight.data.normal_(mean=0.0, std=std)
        self.image_k_proj.bias.data.zero_()
        self.image_v_proj.bias.data.zero_()

        
    def set_image_embeds(self, image_embeds, media_offsets=None, layernorm=None):
        if image_embeds is not None:
            self.image_embeds = layernorm(image_embeds)
            self.media_offsets = media_offsets
        else:
            self.image_embeds = None
            self.media_offsets = None
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        
        if self.image_embeds is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )

        bsz, q_len, _ = hidden_states.size()
        _, image_len, _ = self.image_embeds.size()
        W = H = int(math.sqrt(image_len))

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Shift the token after image to right
        position_ids = position_ids.expand(bsz, -1)
        left_pos_mask = position_ids < self.media_offsets.unsqueeze(1)
        right_pos_mask = ~left_pos_mask
        position_ids = position_ids + right_pos_mask * image_len

        # query_states = self.q_norm(query_states)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        
        if q_len > 1:  # Need image embeddings
            image_key_states = self.image_k_proj(self.image_embeds)
            image_value_states = self.image_v_proj(self.image_embeds)
            image_key_states = image_key_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (bsz, num_key_value_heads, image_len, head_dim)
            image_value_states = image_value_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            new_key_states = torch.zeros(
                (bsz, self.num_key_value_heads, q_len + image_len, self.head_dim),
                device=key_states.device,
                dtype=key_states.dtype,
            )
            new_value_states = torch.zeros_like(new_key_states)
            new_position_ids = torch.zeros((bsz, q_len + image_len), device=position_ids.device, dtype=position_ids.dtype)
            
            for i in range(bsz):
                left_mask = left_pos_mask[i]
                right_mask = right_pos_mask[i]
                media_offset = self.media_offsets[i]

                new_key_states[i] = torch.cat([key_states[i, :, left_mask], image_key_states[i], key_states[i, :, right_mask]], dim=1)
                new_value_states[i] = torch.cat([value_states[i, :, left_mask], image_value_states[i], value_states[i, :, right_mask]], dim=1)
                image_position_id = torch.arange(media_offset, media_offset + image_len, device=position_ids.device, dtype=position_ids.dtype)
                new_position_ids[i] = torch.cat([position_ids[i, left_mask], image_position_id, position_ids[i, right_mask]], dim=0)
                
            key_states = new_key_states
            value_states = new_value_states
            position_ids = new_position_ids

        # key_states = self.k_norm(key_states)
        cos, sin = self.rotary_emb(value_states, position_ids)
        _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        if attention_mask is None and q_len != key_states.size(-2) and q_len > 1:
            attention_mask = torch.triu(torch.ones((bsz, 1, q_len, q_len), dtype=key_states.dtype, device=key_states.device), diagonal=1)
            attention_mask = attention_mask.masked_fill(attention_mask == 1, torch.finfo(attention_mask.dtype).min)

        if attention_mask is not None:
            new_attention_mask = torch.zeros(
                (bsz, 1, q_len, attention_mask.size(-1) + image_len),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            for i in range(bsz):
                left_mask = left_pos_mask[i]
                right_mask = right_pos_mask[i]
                media_offset = self.media_offsets[i]
                attention_mask_i = attention_mask[i, 0]

                if torch.any(left_mask):
                    new_attention_mask[i, :, left_mask] = torch.cat([
                        attention_mask_i[left_mask, :media_offset], 
                        torch.full((left_mask.sum(), image_len), torch.finfo(attention_mask.dtype).min, device=attention_mask.device, dtype=attention_mask.dtype),
                        attention_mask_i[left_mask, media_offset:]
                    ], dim=-1)

                if torch.any(right_mask):
                    new_attention_mask[i, :, right_mask] = torch.cat([
                        attention_mask_i[right_mask, :media_offset], 
                        torch.zeros((right_mask.sum(), image_len), device=attention_mask.device, dtype=attention_mask.dtype),
                        attention_mask_i[right_mask, media_offset:]
                    ], dim=-1)
            attention_mask = new_attention_mask

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class Qwen2FlashAttention2_v2(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_embeds = None
        self.multimodal_gate_proj = nn.Linear(self.head_dim, self.head_dim)
        self.image_key_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.image_value_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # self.layer_proj = nn.Linear(self.hidden_size, self.hidden_size)
        # embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size))
        # self.layer_feature = nn.Parameter(torch.randn((1, 1, self.config.hidden_size)) * embed_std)
        # self.alpha = None
        
        self.valid_image_count = None
        # self.cross = CrossAttentionWithoutValue(self.config)
        
    def set_image_embeds(self, image_embeds, layernorm=None):
        if image_embeds is not None:
            # Padding to same number of images
            self.image_embeds = layernorm(image_embeds)
        else:
            self.image_embeds = None
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,  # This parameter is unused, can be removed if not needed
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if self.image_embeds is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )
        
        output_attentions = False
        
        bsz, q_len, _ = hidden_states.size()
        _, image_len, _ = self.image_embeds.size()
        n_image = 1
        
        # # Retrieve CLS tokens
        # text_cls_token = hidden_states.unsqueeze(-2)  # (bsz, q_len, 1, hidden_size)
        # image_cls_token = self.image_cls_token.unsqueeze(1).expand(-1, q_len, -1, -1)  # (bsz, q_len, n_image, hidden_size)
        # cls_token = torch.cat([image_cls_token, text_cls_token], dim=-2)  # (bsz, q_len, n_image+1, hidden_size)
        # scores = self.cross(text_cls_token, cls_token, self.valid_image_count+1).squeeze(-2)  # (bsz, q_len, n_image+1)
        scores = F.sigmoid(self.multimodal_gate_proj(hidden_states.view(bsz, q_len, self.num_heads, self.head_dim)).view(bsz, q_len, self.hidden_size))
        
        # Project hidden states once
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bsz, num_heads, q_len, head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings to query_states once
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        
        query_states = query_states.transpose(1, 2)
        
        # Project image embeddings once
        image_key_states = self.image_key_proj(self.image_embeds)
        image_value_states = self.image_value_proj(self.image_embeds)
        image_key_states = image_key_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)  # (bsz, num_key_value_heads, image_len, head_dim)
        image_value_states = image_value_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Prepare image_attention_mask once
        if attention_mask is not None:
            image_attention_mask = torch.ones(bsz, image_len, dtype=attention_mask.dtype, device=attention_mask.device)
            # image_attention_mask = torch.cat([image_attention_mask, attention_mask], dim=-1)
        else:
            image_attention_mask = None
            
        # Initialize attn_output
        attn_output = torch.zeros_like(hidden_states)
        dropout_rate = self.attention_dropout if self.training else 0.0
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None
        
        for i in range(n_image + 1):
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                if i == 0:
                    current_key_states, current_value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                else:
                    current_key_states, current_value_states = past_key_value[self.layer_idx]
            else:
                # Clone key_states and value_states to avoid in-place modifications
                current_key_states = key_states.clone()
                current_value_states = value_states.clone()
            
            # Concatenate image key/value states if not the last iteration
            if i != n_image:
                current_key_states = image_key_states  # (bsz, num_key_value_heads, image_len, head_dim)
                current_value_states = image_value_states
                # current_key_states = torch.cat([img_key_states, current_key_states], dim=2)
                # current_value_states = torch.cat([img_value_states, current_value_states], dim=2)
                current_attention_mask = image_attention_mask
            else:
                current_attention_mask = attention_mask
            
            # Apply rotary embeddings to key_states
            pos_ids = torch.arange(current_key_states.size(-2), device=position_ids.device).unsqueeze(0)
            cos, sin = self.rotary_emb(current_value_states, pos_ids)
            _, current_key_states = apply_rotary_pos_emb(current_key_states, current_key_states, cos, sin)
            
            # repeat k/v heads if n_kv_heads < n_heads
            current_key_states = repeat_kv(current_key_states, self.num_key_value_groups)
            current_value_states = repeat_kv(current_value_states, self.num_key_value_groups)
            
            # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
            # to be able to avoid many of these transpose/reshape/view.
            current_key_states = current_key_states.transpose(1, 2)
            current_value_states = current_value_states.transpose(1, 2)
            
            # In PEFT, usually we cast the layer norms in float32 for training stability reasons
            # therefore the input hidden states gets silently casted in float32. Hence, we need
            # cast them back in float16 just to be sure everything works as expected.
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if torch.is_autocast_enabled():
                    target_dtype = torch.get_autocast_gpu_dtype()
                elif hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seem to be silently casted to float32; casting back to {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                current_key_states = current_key_states.to(target_dtype)
                current_value_states = current_value_states.to(target_dtype)
            
            # Compute attention output
            attn_output_result = _flash_attention_forward(
                query_states,
                current_key_states,
                current_value_states,
                current_attention_mask,
                q_len,
                position_ids=pos_ids,
                dropout=dropout_rate,
                sliding_window=sliding_window,
                is_causal=self.is_causal,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
            )
            
            attn_output_result = attn_output_result.reshape(bsz, q_len, self.hidden_size).contiguous()
            attn_output_result = self.o_proj(attn_output_result)
            
            # attn_output.append(attn_output_result)
            # Accumulate weighted attention outputs
            # weight = scores[:, :, i].unsqueeze(-1)
            if i != n_image:
                attn_output += scores * attn_output_result
            else:
                attn_output += (1 - scores) * attn_output_result
        
        # attn_output = torch.stack(attn_output, dim=1)
        # cls_token = attn_output[:, :, 0]
        # scores = self.cross(self.layer_feature.expand(bsz, -1, -1), cls_token, self.valid_image_count+1).squeeze(1)
        # attn_output = (scores.unsqueeze(-1).unsqueeze(-1) * attn_output).sum(dim=1)
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class Qwen2SdpaAttention_v3(Qwen2Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_embeds = None
        self.multimodal_gate_proj = nn.Linear(self.head_dim, self.head_dim)
        self.image_key_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.image_value_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        
    def set_image_embeds(self, image_embeds, layernorm=None):
        if image_embeds is not None:
            self.image_embeds = layernorm(image_embeds)
        else:
            self.image_embeds = None
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
        if self.image_embeds is None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        else:
            attn_output_text, _, _ = super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()
        _, image_len, _ = self.image_embeds.size()
        
        scores = F.sigmoid(self.multimodal_gate_proj(hidden_states.view(bsz, q_len, self.num_heads, self.head_dim)).view(bsz, q_len, self.hidden_size))

        query_states = self.q_proj(hidden_states)
        key_states = self.image_key_proj(self.image_embeds)
        value_states = self.image_value_proj(self.image_embeds)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, image_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cos, sin)
        
        pos_ids = torch.arange(image_len, device=position_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(value_states, pos_ids)
        _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # if attention_mask is not None:
        attention_mask = torch.zeros((bsz, 1, q_len, image_len), dtype=attention_mask.dtype, device=attention_mask.device)
        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output_image = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output_image = attn_output_image.transpose(1, 2).contiguous()
        attn_output_image = attn_output_image.view(bsz, q_len, self.hidden_size)

        attn_output_image = self.o_proj(attn_output_image)
        
        attn_output = scores * attn_output_text + (1 - scores) * attn_output_image

        return attn_output, None, past_key_value
    

class PaidgeQwen2ForCausalLM(LlavaQwenForCausalLM):
    def __init__(self, model: Union[PreTrainedModel, str], **model_init_kwargs):
        if isinstance(model, PretrainedConfig):
            model = LlavaQwenForCausalLM(model)
        
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        if isinstance(model, PreTrainedModel):
            self.__dict__.update(model.__dict__)
        
        # self.target_layers = []
        # self.target_modules = []
        # self.target_input_layernorms = []
        # for i in model.config.mm_cross_select_layer.split(','):
        #     new_layer = Qwen2DecoderLayer(self.config, i).to(self.device)
        #     new_module = Qwen2Attention_v3(
        #         self.config,
        #         i
        #     ).to(self.device)
        #     setattr(new_layer, "self_attn", new_module)
        #     model.model.layers.insert(int(i), new_layer)
        #     self.target_layers.append(new_layer)
        #     self.target_modules.append(new_module)
        #     self.target_input_layernorms.append(new_layer.input_layernorm)
        # for i, layer in enumerate(model.model.layers):
        #     layer.self_attn.layer_idx = i
            
        # self.cross = CrossAttentionWithoutValue(self.config)
        
        self.target_layers = []
        self.target_modules = []
        self.target_input_layernorms = []
        named_modules = [(name, module) for name, module in model.named_modules()]

        for i in model.config.mm_cross_select_layer.split(','):
            parentName = f"model.layers.{i}"
            childName = 'self_attn'
            parent, target = None, None
            for name, module in named_modules:
                if name.endswith(parentName):
                    parent = module
                    target = getattr(parent, childName, None)
                    if target is None:
                        warnings.warn(f"Module {parentName}.{childName} not found")
                        continue
                    if not isinstance(target, Qwen2Attention):
                        warnings.warn(f"Module {parentName}.{childName} is not Qwen2Attention")
                        continue
                    break
            if parent is None or target is None:
                warnings.warn(f"Module {parentName}.{childName} not found")
                continue
            new_module = Qwen2Attention_rework(
                # self.cross,
                target.config,
                target.layer_idx
            )
            new_module = new_module.to(self.device) if self.device.type != "meta" else new_module
            new_module.load_state_dict(target.state_dict(), strict=False)
            if getattr(self.config, "mm_kvproj", False):
                new_module.image_k_proj.load_state_dict(target.k_proj.state_dict(), strict=False)
                new_module.image_v_proj.load_state_dict(target.v_proj.state_dict(), strict=False)
            setattr(parent, childName, new_module)
            self.target_layers.append(parent)
            self.target_modules.append(new_module)
            self.target_input_layernorms.append(parent.input_layernorm)
    
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        images=None,
        image_sizes=None,
        inputs_embeds=None,
        modalities=["image"],
        prompts=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        cache_position=None,
        logits_to_keep=0,
        **loss_kwargs
    ):
        
        if inputs_embeds is None:
            B = input_ids.size(0)
            image_indeces = torch.where(input_ids == IMAGE_TOKEN_INDEX)
            text_indeces = torch.where(input_ids != IMAGE_TOKEN_INDEX)
            input_ids = input_ids[text_indeces[0], text_indeces[1]].view(B, text_indeces[1].size(0) // B)
            attention_mask = attention_mask[text_indeces[0], text_indeces[1]].view(B, text_indeces[1].size(0) // B) if attention_mask is not None else None
            labels = labels[text_indeces[0], text_indeces[1]].view(B, text_indeces[1].size(0) // B) if labels is not None else None
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            
            
        if images is not None:
            global image_attention_mask
            image_attention_mask = None

            for module in self.target_modules:
                module.set_image_embeds(None)
                # module.set_mask(None)
            
            # save images
            # import torchvision.transforms as transforms
            # image_mean = torch.tensor(self.get_model().vision_tower.image_processor.image_mean)  # CLIP Image Mean
            # image_std = torch.tensor((self.get_model().vision_tower.image_processor.image_std))  # CLIP Image Std
            # reverse_normalize = transforms.Normalize(
            #     mean=[-m / s for m, s in zip(image_mean, image_std)],  # Reverse mean
            #     std=[1 / s for s in image_std]  # Reverse std
            # )
            # transform = transforms.Compose([
            #     reverse_normalize,  # Apply reverse normalization
            #     transforms.ToPILImage()  # Convert to PIL image
            # ])
            # image = images[0].float()
            # pil_image = transform(image)
            # pil_image.save("image/original_image.jpg")

            # images_list = []
            # for image in images:
            #     if image.ndim == 4:
            #         images_list.append(image)
            #     else:
            #         images_list.append(image.unsqueeze(0))
                    
            #     concat_images = torch.cat([image for image in images_list], dim=0)
            #     encoded_image_features = self.encode_images(concat_images)
            #     split_sizes = [image.shape[0] for image in images_list]
            #     for i, (module, input_norm) in enumerate(zip(self.target_modules, self.target_input_layernorms)):
            #         module.set_image_embeds(input_norm(encoded_image_features), split_sizes)
            
            images = torch.stack(images, dim=0)
            image_embeds = self.encode_images(images)
            
            if image_embeds.ndim == 4:
                # This should not happen
                # raise ValueError("Image embeddings should have 3 dimensions")
                # droplist = [random.choices([True, False], weights=[0.8, 0.2], k=1)[0] for _ in range(image_embeds.size(0))]
                # droplist[-1] = False
                # for i, (image_embed, module, input_norm, drop) in enumerate(zip(image_embeds, self.target_modules, self.target_input_layernorms, droplist)):
                #     module.set_image_embeds(image_embed.detach() if drop else image_embed, image_indeces[-1], input_norm)
                for i, (image_embed, module, input_norm) in enumerate(zip(image_embeds, self.target_modules, self.target_input_layernorms)):
                    module.set_image_embeds(image_embed, image_indeces[-1], input_norm)
            elif image_embeds.ndim == 3:
                # This should not happen
                raise ValueError("Image embeddings should have 4 dimensions")
                for i, (module, input_norm) in enumerate(zip(self.target_modules, self.target_input_layernorms)):
                    module.set_image_embeds(image_embeds, image_indeces[-1], input_norm)
            
            # # Encode images once
            # images_list = []
            # split_sizes = []

            # # Collect valid images and handle None cases
            # for image in images:
            #     if image.ndim == 4:
            #         images_list.append(image)
            #         split_sizes.append(image.shape[0])  # Add the batch size of the image
            #     else:
            #         images_list.append(image.unsqueeze(0))
            #         split_sizes.append(1)  # If we unsqueeze, it's 1 image in batch

            # if len(images_list) != 0:
            #     concat_images = torch.cat([image for image in images_list], dim=0)
            #     encoded_image_features = self.encode_images(concat_images)
                
            #     if encoded_image_features.ndim == 4:
            #         for i, (image_feature, module, input_norm) in enumerate(zip(encoded_image_features, self.target_modules, self.target_input_layernorms)):
            #             module.set_image_embeds(image_feature, split_sizes, input_norm)
            #     elif encoded_image_features.ndim == 3:
            #         for i, (module, input_norm) in enumerate(zip(self.target_modules, self.target_input_layernorms)):
            #             module.set_image_embeds(encoded_image_features, split_sizes, input_norm)
            
        return super().forward(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **loss_kwargs
        )
    
    @torch.no_grad()
    def generate(self, input_ids, *args, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        modalities = kwargs.pop("modalities", ["image"])
        
        B = input_ids.size(0)
        image_indeces = torch.where(input_ids == IMAGE_TOKEN_INDEX)
        text_indeces = torch.where(input_ids != IMAGE_TOKEN_INDEX)
        input_ids = input_ids[text_indeces[0], text_indeces[1]].view(B, text_indeces[1].size(0) // B)
        inputs_embeds = self.get_model().embed_tokens(input_ids)
        
        kwargs['inputs_embeds'] = inputs_embeds

        for module in self.target_modules:
            global image_attention_mask
            image_attention_mask = None
            module.set_image_embeds(None)
            # module.set_mask(None)
            
        image_embeds = self.encode_images(images)
        
        if image_embeds.ndim == 4:
            # This should not happen
            # raise ValueError("Image embeddings should have 3 dimensions")
            for i, (image_embed, module, input_norm) in enumerate(zip(image_embeds, self.target_modules, self.target_input_layernorms)):
                module.set_image_embeds(image_embed, image_indeces[-1], input_norm)
        elif image_embeds.ndim == 3:
            # This should not happen
            # raise ValueError("Image embeddings should have 4 dimensions")
            for i, (module, input_norm) in enumerate(zip(self.target_modules, self.target_input_layernorms)):
                module.set_image_embeds(image_embeds, image_indeces[-1], input_norm)

        # # Encode images once
        # images_list = []
        # split_sizes = []

        # # Collect valid images and handle None cases
        # for image in images:
        #     if image.ndim == 4:
        #         images_list.append(image)
        #         split_sizes.append(image.shape[0])  # Add the batch size of the image
        #     else:
        #         images_list.append(image.unsqueeze(0))
        #         split_sizes.append(1)  # If we unsqueeze, it's 1 image in batch

        # if len(images_list) != 0:
        #     concat_images = torch.cat([image for image in images_list], dim=0)
        #     encoded_image_features = self.encode_images(concat_images)
        #     if encoded_image_features.ndim == 4:
        #         for i, (image_feature, module, input_norm) in enumerate(zip(encoded_image_features, self.target_modules, self.target_input_layernorms)):
        #             module.set_image_embeds(F.gelu(image_feature), split_sizes, input_norm)
        #     elif encoded_image_features.ndim == 3:
        #         for i, (module, input_norm) in enumerate(zip(self.target_modules, self.target_input_layernorms)):
        #             module.set_image_embeds(F.gelu(encoded_image_features), split_sizes, input_norm)
        
        return super().generate(
            inputs = input_ids,
            images = None,
            image_sizes = None,
            **kwargs
        )
    
    # def save_pretrained(
    #     self,
    #     save_directory: Union[str, os.PathLike],
    #     is_main_process: bool = True,
    #     state_dict: Optional[dict] = None,
    #     save_function: Callable = torch.save,
    #     push_to_hub: bool = False,
    #     max_shard_size: Union[int, str] = "5GB",
    #     safe_serialization: bool = True,
    #     variant: Optional[str] = None,
    #     token: Optional[Union[str, bool]] = None,
    #     save_peft_format: bool = True,
    #     **kwargs,
    # ):
    #     # Save Model
    #     super().save_pretrained(
    #         save_directory,
    #         is_main_process,
    #         state_dict,
    #         save_function,
    #         push_to_hub,
    #         max_shard_size,
    #         safe_serialization,
    #         variant,
    #         token, 
    #         save_peft_format,
    #         **kwargs
    #     )
        
    #     # Save config
    #     self.config.save_pretrained(save_directory)
    #     # Save meta block
    #     # torch.save(self.meta_block.state_dict(), os.path.join(save_directory, 'meta_block.bin'))
    #     # Save vision towel
    #     non_lora_state_dict = {k: t for k, t in self.state_dict().items() if "lora_" not in k and 'meta_' not in k}
    #     torch.save(non_lora_state_dict, os.path.join(save_directory, 'non_lora_trainables.bin'))
    
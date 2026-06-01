import torch
from torch import nn
import torch.distributed as dist
import torch.nn.attention.flex_attention
import torch.nn.functional as F
from typing import Any, Optional, Union, Callable, Tuple
import warnings
import os
import math
from abc import ABC, abstractmethod
import random

from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from transformers.models.llama.modeling_llama import eager_attention_forward, ALL_ATTENTION_FUNCTIONS, LlamaRotaryEmbedding, LlamaAttention, LlamaRMSNorm, apply_rotary_pos_emb, repeat_kv, rotate_half, logger
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.activations import ACT2FN
from transformers import CLIPModel, SiglipModel, AutoTokenizer

from llava_next.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava_next.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava_next.model.multimodal_encoder.clip_encoder import CLIPTextTower
from llava_next.model.multimodal_projector.builder import build_text_projector

from llava_next.utils import rank0_print


class LlamaRotaryEmbedding_3D(LlamaRotaryEmbedding):
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

class SimpleResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(in_channels)

        self.mlp = nn.Sequential(nn.Linear(in_channels, out_channels), nn.GELU(), nn.Linear(out_channels, out_channels))

        self.proj = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x_norm = self.pre_norm(x)
        x_mlp = self.mlp(x_norm)
        x_proj = self.proj(x_norm)
        return x_proj + x_mlp

image_attention_mask = None

class LlamaAttention_rework(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_embeds = None
        self.media_offsets = None
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=self.config.rms_norm_eps) if getattr(self.config, "mm_qknorm", False) else nn.Identity()
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=self.config.rms_norm_eps) if getattr(self.config, "mm_qknorm", False) else nn.Identity()
        
        modules = []
        if getattr(self.config, "mm_proj_type", None) is None:
            modules.append(nn.Linear(self.config.mm_hidden_size, self.config.hidden_size))
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
        else:
            if "ln" in self.config.mm_proj_type:
                modules.append(nn.LayerNorm(self.config.mm_hidden_size, eps=self.config.rms_norm_eps))
            if "linear" in self.config.mm_proj_type:
                modules.append(nn.Linear(self.config.mm_hidden_size, self.config.hidden_size))
            if "mlp" in self.config.mm_proj_type:
                modules.append(nn.Linear(self.config.mm_hidden_size, self.config.hidden_size))
                modules.append(nn.GELU())
                modules.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
            if "res" in self.config.mm_proj_type:
                # modules.append(nn.Linear(self.config.mm_hidden_size, self.config.hidden_size))
                modules.append(SimpleResBlock(self.config.mm_hidden_size, self.config.hidden_size))

        self.mm_proj = nn.Sequential(*modules)

        # Initialize the mm_proj layers
        for module in self.mm_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

        # self.drop_image = nn.Dropout(self.config.mm_proj_dropout)
            
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.rotary_emb_2D = LlamaRotaryEmbedding_3D(config=self.config)
        if getattr(self.config, "mm_kvproj", False):
            self.image_k_proj = nn.Linear(self.config.hidden_size, self.config.num_key_value_heads * self.head_dim, bias=self.config.attention_bias)
            self.image_v_proj = nn.Linear(self.config.hidden_size, self.config.num_key_value_heads * self.head_dim, bias=self.config.attention_bias)

        
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
        if getattr(self.config, "mm_rope", False) is True:
            position_ids = position_ids + right_pos_mask * image_side_len
        else:
            position_ids = position_ids + right_pos_mask * image_len
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
            if getattr(self.config, "mm_rope", False) is True:
                image_position_id = torch.stack([
                    torch.arange(0, image_side_len, device=position_ids.device, dtype=position_ids.dtype).repeat_interleave(image_side_len), 
                    torch.arange(0, image_side_len, device=position_ids.device, dtype=position_ids.dtype).tile(image_side_len), 
                    torch.full((image_len,), image_side_len//2, device=position_ids.device, dtype=position_ids.dtype)
                ], dim=0)
            else:
                image_position_id = torch.arange(0, image_len, device=position_ids.device, dtype=position_ids.dtype)


            new_key_states = torch.empty(
                (bsz, self.config.num_key_value_heads, q_len + image_len, self.head_dim),
                device=key_states.device,
                dtype=key_states.dtype,
            )
            new_value_states = torch.empty_like(new_key_states)
            if getattr(self.config, "mm_rope", False) is True:
                new_position_ids = torch.empty((3, bsz, q_len + image_len), device=position_ids.device, dtype=position_ids.dtype)
            else:
                new_position_ids = torch.empty((bsz, q_len + image_len), device=position_ids.device, dtype=position_ids.dtype)

            # Precompute expanded indices for efficiency
            text_indices_expanded = text_indices[:, None, :, None].expand(bsz, self.config.num_key_value_heads, -1, self.head_dim)
            image_indices_expanded = image_indices[:, None, :, None].expand(bsz, self.config.num_key_value_heads, -1, self.head_dim)

            # Perform scatter_ in a more optimized way
            new_key_states.scatter_(2, text_indices_expanded, key_states)
            new_key_states.scatter_(2, image_indices_expanded, image_key_states)

            new_value_states.scatter_(2, text_indices_expanded, value_states)
            new_value_states.scatter_(2, image_indices_expanded, image_value_states)

            if getattr(self.config, "mm_rope", False) is True:
                # Precompute position IDs expansion (3D indices)
                text_position_ids_expanded = position_ids.expand(3, bsz, -1)
                image_position_id_expanded = image_position_id[:, None].expand(3, bsz, -1) + self.media_offsets[:, None]

                # Scatter text and image positions into the new tensor
                new_position_ids.scatter_(2, text_indices.expand(3, bsz, -1), text_position_ids_expanded)
                new_position_ids.scatter_(2, image_indices.expand(3, bsz, -1), image_position_id_expanded)
            
            else:
                # Precompute position IDs expansion (2D indices)
                text_position_ids_expanded = position_ids.expand(bsz, -1)
                image_position_id_expanded = image_position_id[None, :].expand(bsz, -1) + self.media_offsets[:, None]

                # Scatter text and image positions into the new tensor
                new_position_ids.scatter_(1, text_indices.expand(bsz, -1), text_position_ids_expanded)
                new_position_ids.scatter_(1, image_indices.expand(bsz, -1), image_position_id_expanded)

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
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        # self.attn_weights = attn_weights
        return attn_output, attn_weights
    

class PaidgeLlamaForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, model: Union[PreTrainedModel, str], **model_init_kwargs):
        if isinstance(model, PretrainedConfig):
            model = LlavaLlamaForCausalLM(model)
        
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
                    if not isinstance(target, LlamaAttention):
                        warnings.warn(f"Module {parentName}.{childName} is not LlamaAttention")
                        continue
                    break
            if parent is None or target is None:
                warnings.warn(f"Module {parentName}.{childName} not found")
                continue
            new_module = LlamaAttention_rework(
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
        
        self.vision_text_model = None

    def initialize_text_modules(self, model_args, fsdp=None):
        if 'clip' in model_args.vision_tower.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_args.vision_tower)
            mm_tower = CLIPModel.from_pretrained(
                model_args.vision_tower,
                attn_implementation="flash_attention_2",
                device_map='cuda'
            )
        elif 'siglip' in model_args.vision_tower.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_args.vision_tower)
            mm_tower = SiglipModel.from_pretrained(
                model_args.vision_tower,
                attn_implementation="flash_attention_2",
                device_map='cuda'
            )
        else:
            raise ValueError(f"Unsupported vision tower: {model_args.vision_tower}")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.

        del mm_tower.vision_model
        mm_tower.vision_model = self.get_model().get_vision_tower().vision_tower.vision_model
        self.alt_tokenizer = tokenizer
        mm_tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.mm_tower = mm_tower
    
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
            new_input_ids = input_ids[text_indeces[0], text_indeces[1]].view(B, -1)
            attention_mask = attention_mask[text_indeces[0], text_indeces[1]].view(B, -1) if attention_mask is not None else None
            labels = labels[text_indeces[0], text_indeces[1]].view(B, -1) if labels is not None else None
            inputs_embeds = self.get_model().embed_tokens(new_input_ids)
            
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
            #             module.set_image_embeds(image_feature, split_sizes, input_norm)
            #     elif encoded_image_features.ndim == 3:
            #         for i, (module, input_norm) in enumerate(zip(self.target_modules, self.target_input_layernorms)):
            #             module.set_image_embeds(encoded_image_features, split_sizes, input_norm)

        # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
        to_return = super().forward(
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

        if getattr(self, "mm_tower", None) is None:
            return to_return
            
        self.prompts.extend(prompts)
        self.images.append(images)
        self.current_accumulation_step += 1
        
        if self.current_accumulation_step == 1:
            self.LLM_loss = []

        self.LLM_loss.append(to_return[0].detach())

        if self.current_accumulation_step == self.gradient_accumulation_steps:
            # Padding prompts
            batch_prompts = {}
            batch_prompts["input_ids"] = torch.nn.utils.rnn.pad_sequence(self.prompts, batch_first=True, padding_value=self.alt_tokenizer.pad_token_id)
            batch_prompts["attention_mask"] = batch_prompts["input_ids"].ne(self.alt_tokenizer.pad_token_id)

            batch_images = torch.cat(self.images, dim=0)
            alt_loss = self.forward_mm_tower(batch_prompts, batch_images)

            self.prompts = []
            self.images = []
            self.current_accumulation_step = 0

            to_return = list(to_return)
            to_return[0] = to_return[0] + alt_loss
            to_return = tuple(to_return)

            self.LLM_loss = torch.sum(torch.stack(self.LLM_loss))
            self.alt_loss = alt_loss.detach()

        return to_return
    

    def forward_mm_tower(
        self,
        prompts,
        images
    ):
        model = self.mm_tower
        text_embeds = model.get_text_features(**prompts)
        image_embeds = model.get_image_features(images)
        num_samples = image_embeds.shape[0]

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()

        # Calculate the Loss for text image alignment
        if isinstance(model, CLIPModel):
            all_image_features = torch.cat(dist.nn.all_gather(image_embeds), dim=0)
            all_text_features = torch.cat(dist.nn.all_gather(text_embeds), dim=0)

            logits_per_text = torch.matmul(text_embeds, all_image_features.t()) * logit_scale
            logits_per_image = torch.matmul(image_embeds, all_text_features.t()) * logit_scale

            # Get rank
            rank = dist.get_rank()
            labels = torch.arange(num_samples, device=logits_per_image.device, dtype=torch.long)
            labels = labels + num_samples * rank

            loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

            loss = loss * 30

        elif isinstance(model, SiglipModel):

            def siglip_loss(image_features, text_features, labels):
                logits_per_text = (
                    torch.matmul(image_features, text_features.t()) * logit_scale
                    + model.logit_bias
                )
                loglik = F.logsigmoid(labels * logits_per_text)
                nll = -torch.sum(loglik, dim=-1)
                loss = nll.mean()
                return loss

            all_text = dist.nn.all_gather(text_embeds)
            rank = dist.get_rank()

            loss = 0
            for i, gathered_txt in enumerate(all_text):
                labels = -torch.ones((num_samples, num_samples), device=images.device)
                if i == rank:
                    labels += 2 * torch.eye(num_samples, device=images.device)
                loss += siglip_loss(image_embeds, gathered_txt, labels)
            
            loss = loss * 1
        
        else:
            raise ValueError(f"Unsupported vision tower: {model}")

        return loss / num_samples

    
    @torch.no_grad()
    def generate(self, input_ids, *args, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        modalities = kwargs.pop("modalities", ["image"])
        
        B = input_ids.size(0)
        image_indeces = torch.where(input_ids == IMAGE_TOKEN_INDEX)
        text_indeces = torch.where(input_ids != IMAGE_TOKEN_INDEX)
        input_ids = input_ids[text_indeces[0], text_indeces[1]].view(B, -1)
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
    
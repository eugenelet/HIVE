import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Union, Callable, Tuple
import warnings
import os
import math
from abc import ABC, abstractmethod

from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2FlashAttention2, Qwen2RMSNorm, apply_rotary_pos_emb, repeat_kv, logger
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.activations import ACT2FN

from llava_next.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
from llava_next.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

from llava_next.utils import rank0_print


class Qwen2DecoderLayer_v2(Qwen2DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2FlashAttention2_v2(config, layer_idx)
        self.alpha_xattn = nn.Parameter(torch.tensor(0.0))
        self.alpha_dense = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states * self.alpha_xattn.tanh()

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.alpha_dense.tanh()

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen2FlashAttention2_v2(Qwen2FlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
                self.valid_image_count = torch.tensor(valid_image_count, dtype=torch.long, device=image_embeds.device)
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
        
        query_states = query_states.transpose(1, 2)
        
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
        dropout_rate = self.attention_dropout if self.training else 0.0
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None
        
        for i in range(n_image):
            # Concatenate image key/value states if not the last iteration
            current_key_states = image_key_states[:, i]  # (bsz, num_key_value_heads, image_len, head_dim)
            current_value_states = image_value_states[:, i]
            
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
                None,
                q_len,
                position_ids=pos_ids,
                dropout=dropout_rate,
                sliding_window=sliding_window,
                is_causal=False,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
            )
            
            attn_output_result = attn_output_result.reshape(bsz, q_len, self.hidden_size).contiguous()
            attn_output_result = self.o_proj(attn_output_result)
            
            attn_output = attn_output_result
        
        if not output_attentions:
            attn_weights = None
        
        return attn_output, attn_weights, past_key_value


class FlamingoModel(LlavaQwenForCausalLM):
    def __init__(self, model: Union[PreTrainedModel, str, ], **model_init_kwargs):
        if isinstance(model, PretrainedConfig):
            model = LlavaQwenForCausalLM(model)
        
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        if isinstance(model, PreTrainedModel):
            self.__dict__.update(model.__dict__)
        
        self.target_layers = []
        self.target_modules = []
        self.target_input_layernorms = []
        for i in range(0, self.config.num_hidden_layers*2 , 2):  
            # new_layer = Qwen2DecoderLayer_v2(self.config, 0).to(self.device)
            new_layer = Qwen2DecoderLayer_v2(self.config, 0)
            if self.device != torch.device("meta"):
                new_layer = new_layer.to(self.device)
            self.model.layers.insert(int(i), new_layer)
            self.target_layers.append(new_layer)
            self.target_modules.append(new_layer.self_attn)
            self.target_input_layernorms.append(new_layer.input_layernorm)
        # for i, layer in enumerate(model.model.layers):
        #     layer.self_attn.layer_idx = i
        
        
    def forward(self, *args, **kwargs):       
        input_ids = kwargs.pop("input_ids", None)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        past_key_values = kwargs.pop("past_key_values", None)
        labels = kwargs.pop("labels", None)
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        modalities = kwargs.pop("modalities", ["image"])
        prompts = kwargs.pop("prompts", None)
        
        if inputs_embeds is None:
            # B = input_ids.size(0)
            # text_indeces = torch.where(input_ids != -200)
            # input_ids = input_ids[text_indeces[0], text_indeces[1]].view(B, text_indeces[1].size(0) // B)
            # attention_mask = attention_mask[text_indeces[0], text_indeces[1]].view(B, text_indeces[1].size(0) // B) if attention_mask is not None else None
            # labels = labels[text_indeces[0], text_indeces[1]].view(B, text_indeces[1].size(0) // B) if labels is not None else None
            # inputs_embeds = self.get_model().embed_tokens(input_ids)
            
            # Remove IMAGE_TOKEN_INDEX from each sequence
            B = input_ids.size(0)
            mask = input_ids != IMAGE_TOKEN_INDEX  # [B, L]
            lengths = mask.sum(dim=1)  # [B]
            max_length = lengths.max().item()
            
            # Find sequences without IMAGE_TOKEN_INDEX
            has_image_token = (input_ids == IMAGE_TOKEN_INDEX).any(dim=1)  # [B], True if sequence has IMAGE_TOKEN_INDEX

            # Set images[i] = None where there is no IMAGE_TOKEN_INDEX
            images = [images[i] if has_image_token[i] else None for i in range(B)]

            # Prepare new tensors
            new_input_ids = torch.zeros((B, max_length), dtype=input_ids.dtype, device=input_ids.device)
            new_attention_mask = torch.zeros((B, max_length), dtype=attention_mask.dtype, device=attention_mask.device) if attention_mask is not None else None
            new_labels = torch.zeros((B, max_length), dtype=labels.dtype, device=labels.device) if labels is not None else None

            for b in range(B):
                seq_len = lengths[b]
                new_input_ids[b, :seq_len] = input_ids[b][mask[b]]
                if attention_mask is not None:
                    new_attention_mask[b, :seq_len] = attention_mask[b][mask[b]]
                if labels is not None:
                    new_labels[b, :seq_len] = labels[b][mask[b]]

            input_ids = new_input_ids
            attention_mask = new_attention_mask
            labels = new_labels
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            
            
        if images is not None:
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
            
            # Encode images once
            images_list = []
            split_sizes = []

            # Collect valid images and handle None cases
            for image in images:
                if image is not None:
                    if image.ndim == 4:
                        images_list.append(image)
                        split_sizes.append(image.shape[0])  # Add the batch size of the image
                    else:
                        images_list.append(image.unsqueeze(0))
                        split_sizes.append(1)  # If we unsqueeze, it's 1 image in batch
                else:
                    split_sizes.append(0)  # For None, we add 0 to split_sizes

            if len(images_list) != 0:
                concat_images = torch.cat([image for image in images_list], dim=0)
                encoded_image_features = self.encode_images(concat_images)
                
                if encoded_image_features.ndim == 4:
                    for i, (image_feature, module, input_norm) in enumerate(zip(encoded_image_features, self.target_modules, self.target_input_layernorms)):
                        module.set_image_embeds(input_norm(image_feature), split_sizes)
                elif encoded_image_features.ndim == 3:
                    for i, (module, input_norm) in enumerate(zip(self.target_modules, self.target_input_layernorms)):
                        module.set_image_embeds(input_norm(encoded_image_features), split_sizes)
                
        # kwargs["input_ids"] = input_ids
        kwargs["position_ids"] = position_ids
        kwargs["attention_mask"] = attention_mask
        kwargs["past_key_values"] = past_key_values
        kwargs["inputs_embeds"] = inputs_embeds
        kwargs["labels"] = labels
            
        return super().forward(*args, **kwargs)
    
    @torch.no_grad()
    def generate(self, input_ids, *args, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        modalities = kwargs.pop("modalities", ["image"])
        
        # Remove IMAGE_TOKEN_INDEX from each sequence
        B = input_ids.size(0)
        mask = input_ids != IMAGE_TOKEN_INDEX  # [B, L]
        lengths = mask.sum(dim=1)  # [B]
        max_length = lengths.max().item()
        
        # Find sequences without IMAGE_TOKEN_INDEX
        has_image_token = (input_ids == IMAGE_TOKEN_INDEX).any(dim=1)  # [B], True if sequence has IMAGE_TOKEN_INDEX

        # Set images[i] = None where there is no IMAGE_TOKEN_INDEX
        images = [images[i] if has_image_token[i] else None for i in range(B)]

        # Prepare new tensors
        new_input_ids = torch.zeros((B, max_length), dtype=input_ids.dtype, device=input_ids.device)

        for b in range(B):
            seq_len = lengths[b]
            new_input_ids[b, :seq_len] = input_ids[b][mask[b]]

        input_ids = new_input_ids
        inputs_embeds = self.get_model().embed_tokens(input_ids)
        
        kwargs['inputs_embeds'] = inputs_embeds

        for module in self.target_modules:
            module.set_image_embeds(None)
            # module.set_mask(None)

        # Encode images once
        images_list = []
        split_sizes = []

        # Collect valid images and handle None cases
        for image in images:
            if image is not None:
                if image.ndim == 4:
                    images_list.append(image)
                    split_sizes.append(image.shape[0])  # Add the batch size of the image
                else:
                    images_list.append(image.unsqueeze(0))
                    split_sizes.append(1)  # If we unsqueeze, it's 1 image in batch
            else:
                split_sizes.append(0)  # For None, we add 0 to split_sizes

        if len(images_list) != 0:
            concat_images = torch.cat([image for image in images_list], dim=0)
            encoded_image_features = self.encode_images(concat_images)
            if encoded_image_features.ndim == 4:
                for i, (image_feature, module, input_norm) in enumerate(zip(encoded_image_features, self.target_modules, self.target_input_layernorms)):
                    module.set_image_embeds(F.gelu(image_feature), input_norm, split_sizes)
            elif encoded_image_features.ndim == 3:
                for i, (module, input_norm) in enumerate(zip(self.target_modules, self.target_input_layernorms)):
                    module.set_image_embeds(input_norm(encoded_image_features), split_sizes)
            
        return super().generate(
            inputs = input_ids,
            images = None,
            image_sizes = None,
            **kwargs
        )
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        # Save Model
        super().save_pretrained(
            save_directory,
            is_main_process,
            state_dict,
            save_function,
            push_to_hub,
            max_shard_size,
            safe_serialization,
            variant,
            token, 
            save_peft_format,
            **kwargs
        )
        
        # Save config
        self.config.save_pretrained(save_directory)
        # Save meta block
        # torch.save(self.meta_block.state_dict(), os.path.join(save_directory, 'meta_block.bin'))
        # Save vision towel
        non_lora_state_dict = {k: t for k, t in self.state_dict().items() if "lora_" not in k and 'meta_' not in k}
        torch.save(non_lora_state_dict, os.path.join(save_directory, 'non_lora_trainables.bin'))
    
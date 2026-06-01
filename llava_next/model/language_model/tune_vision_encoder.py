import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Optional, Union, Callable, Tuple
import warnings
import os
import math
from abc import ABC, abstractmethod
import functools

from transformers import PreTrainedModel, AutoModelForCausalLM, PretrainedConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2FlashAttention2, Qwen2Attention, Qwen2RMSNorm, apply_rotary_pos_emb, repeat_kv, logger
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava_next.model.language_model.llava_qwen import LlavaQwenForCausalLM
from llava_next.model.language_model.paidge_qwen import PaidgeModelV3ForCausalLM
from llava_next.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava_next.model.multimodal_encoder.clip_encoder import CLIPTextTower
from llava_next.model.multimodal_projector.builder import build_text_projector

from llava_next.utils import rank0_print

class TuneVisionEncoderForCausalLM(LlavaQwenForCausalLM):
    def __init__(self, model: Union[PreTrainedModel, str], **model_init_kwargs):
        if isinstance(model, PretrainedConfig):
            model = LlavaQwenForCausalLM(model)
        
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        if isinstance(model, PreTrainedModel):
            self.__dict__.update(model.__dict__)
            
        if hasattr(self.config, "mm_hidden_size"):
            # # delay_load = getattr(self.config, "delay_load", False)
            # self.text_tower = self.model.embed_tokens
            # # self.text_tower = CLIPTextTower(self.config.mm_text_tower, self.config, delay_load=delay_load)
            # self.mm_text_projector = build_text_projector(self.config)
            # self.get_model().get_vision_tower().text_pos_embedding = nn.Embedding(
            #     77,
            #     self.config.mm_hidden_size,
            #     device=self.device,
            # )
            self.im_head = nn.Linear(
                self.config.hidden_size,
                3 * self.get_model().get_vision_tower().vision_tower.config.patch_size * self.get_model().get_vision_tower().vision_tower.config.patch_size,
            )
            std = self.config.initializer_range
            self.im_head.weight.data.normal_(mean=0.0, std=std)
            self.im_head.bias.data.zero_()
    
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        super().initialize_vision_tokenizer(model_args, tokenizer)
        self.im_head = nn.Linear(
            self.config.hidden_size,
            3 * self.get_model().get_vision_tower().vision_tower.config.patch_size * self.get_model().get_vision_tower().vision_tower.config.patch_size,
        )
        std = self.config.initializer_range
        self.im_head.weight.data.normal_(mean=0.0, std=std)
        self.im_head.bias.data.zero_()

            
    def initialize_text_modules(self, model_args, fsdp=None):
        return
        # mm_text_select_layer = model_args.mm_vision_select_layer
        # pretrain_text_mlp_adapter = model_args.pretrain_text_mlp_adapter

        # # self.config.mm_text_tower = model_args.text_tower

        # # if getattr(self, "text_tower", None) is None:
        #     # text_tower = CLIPTextTower(model_args.text_tower, model_args, delay_load=False)

        # #     if fsdp is not None and len(fsdp) > 0:
        # #         self.text_tower = [text_tower]
        # #     else:
        # #         self.text_tower = text_tower
        # # else:
        # #     if fsdp is not None and len(fsdp) > 0:
        # #         text_tower = self.text_tower[0]
        # #     else:
        # #         text_tower = self.text_tower
        # #     text_tower.load_model()
        # self.text_tower = self.model.embed_tokens
        
        # self.get_model().get_vision_tower().text_pos_embedding = nn.Embedding(
        #     77,
        #     self.get_model().get_vision_tower().hidden_size,
        #     device=self.device,
        # )

        # self.config.use_mm_proj = True
        # self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        # self.config.mm_text_hidden_size = self.config.hidden_size
        # self.config.mm_text_select_layer = mm_text_select_layer

        # if getattr(self, "mm_text_projector", None) is None:
        #     self.mm_text_projector = build_text_projector(self.config)
        # else:
        #     # In case it is frozen by LoRA
        #     for p in self.mm_text_projector.parameters():
        #         p.requires_grad = True

        # if pretrain_text_mlp_adapter is not None:
        #     mm_text_projector_weights = torch.load(pretrain_text_mlp_adapter, map_location="cpu", weights_only=True)

        #     def get_w(weights, keyword):
        #         return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

        #     incompatible_keys = self.mm_text_projector.load_state_dict(get_w(mm_text_projector_weights, "mm_text_projector"))
        #     rank0_print(f"Loaded text projector weights from {pretrain_text_mlp_adapter}. Incompatible keys: {incompatible_keys}")

            
    # def encode_texts(self, **inputs):
    #     # text_features, attention_mask = self.text_tower(inputs)
    #     text_features, attention_mask = self.text_tower(inputs['input_ids']), inputs['attention_mask']
    #     text_features = self.mm_text_projector(text_features)
    #     return text_features, attention_mask
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        # image_features = image_scores.unsqueeze(-1) * image_features
        return image_features
    
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
        num_logits_to_keep=0,
        gradient_accumulation_steps=1,
        **loss_kwargs
    ):  
        
        # if prompts is not None:
        #     # prompt_input_ids = torch.cat([prompt['input_ids'].to(self.device) for prompt in prompts], dim=0)
        #     # prompt_attention_mask = torch.cat([prompt['attention_mask'].to(self.device) for prompt in prompts], dim=0)
        #     text_features, text_attention_mask = self.encode_texts(input_ids=prompts['input_ids'], attention_mask=prompts["attention_mask"])
                
        #     self.get_model().vision_tower.modify_text_features(text_features, text_attention_mask)
        #     # self.get_model().vision_tower.modify_text_features(global_text_features)
            
        
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
        
        # return super().forward(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     labels=labels,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     images=images,
        #     image_sizes=image_sizes,
        #     return_dict=return_dict,
        #     modalities=modalities,
        #     cache_position=cache_position,
        #     num_logits_to_keep=num_logits_to_keep,
        #     **loss_kwargs
        # )
            

        bsz, seq_len, hidden_size = inputs_embeds.size()
        image_idx = labels.eq(IMAGE_TOKEN_INDEX)
        text_idx = labels.ne(IMAGE_TOKEN_INDEX)
        
        images = torch.stack(images)  # (B, C, H, W)
        patch_size = self.get_model().get_vision_tower().vision_tower.config.patch_size 
        # Rearrange the tensor to split it into patches
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        # Reshape to (B, N_h, N_w, C, patch_size, patch_size) to (B, N, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(bsz, -1, 3, patch_size, patch_size)
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            text_labels = labels[text_idx]
            text_logits = logits[text_idx]
            text_loss = self.loss_function(text_logits, text_labels, self.vocab_size, **loss_kwargs)
            
            # TODO: Multiple image token / AnyRes support, currently has gradient accumulation issues
            image_states = hidden_states[image_idx].reshape(bsz, -1, hidden_size)
            image_logits = self.im_head(image_states).reshape(bsz, -1, 3, patch_size, patch_size)
            image_labels = patches
            # Shift so that tokens < n predict n
            shift_image_logits = image_logits[:, :-1].contiguous()
            shift_image_labels = image_labels[:, 1:].contiguous()
            # Flatten the tokens
            shift_image_logits = shift_image_logits.view(-1, 3, patch_size, patch_size)
            shift_image_labels = shift_image_labels.view(-1, 3, patch_size, patch_size)
            # Enable model parallelism
            shift_image_labels = shift_image_labels.to(shift_image_logits.device)
            image_loss = F.mse_loss(shift_image_logits, shift_image_labels, reduction="mean") / gradient_accumulation_steps
            loss = text_loss + 0.2 * image_loss
            # print(f"Text loss: {text_loss.item()}, Image loss: {image_loss.item()}")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @torch.no_grad()
    def generate(self, input_ids, *args, **kwargs):
        prompts = kwargs.pop("prompts", None)
        
        if prompts is not None:
            # prompt_input_ids = torch.cat([prompt['input_ids'].to(self.device) for prompt in prompts], dim=0)
            # prompt_attention_mask = torch.cat([prompt['attention_mask'].to(self.device) for prompt in prompts], dim=0)
            text_features, text_attention_mask = self.encode_texts(input_ids=prompts['input_ids'], attention_mask=prompts["attention_mask"])
                   
            self.get_model().vision_tower.modify_text_features(text_features, text_attention_mask)
            # self.get_model().vision_tower.modify_text_features(global_text_features)
        
        else:
            self.get_model().vision_tower.modify_text_features(None, None)
            print("No prompts, fall back to default behavior")
        
        return super().generate(
            inputs = input_ids,
            *args,
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
    
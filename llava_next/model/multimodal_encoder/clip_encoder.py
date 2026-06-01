import torch
import torch.nn as nn
from llava_next.utils import rank0_print
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPImageProcessorFast, CLIPVisionConfig, CLIPTextConfig, CLIPTextModel, AutoTokenizer
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
import functools

try:
    from s2wrapper import forward as multiscale_forward
except:
    pass

class CLIPTextTower(nn.Module):
    def __init__(self, text_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.text_tower_name = text_tower
        self.select_layer = args.mm_text_select_layer

        if not delay_load:
            rank0_print(f"Loading text tower: {text_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_text_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `text_tower` weights: `unfreeze_mm_text_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_text_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `text_tower` weights: `mm_tunable_parts` contains `mm_text_tower`.")
            self.load_model()
        else:
            self.cfg_only = CLIPTextConfig.from_pretrained(self.text_tower_name)

    def load_model(self, device_map='cuda'):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.text_tower_name))
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.text_tower_name)
        self.text_tower = CLIPTextModel.from_pretrained(
            self.text_tower_name, 
            attn_implementation="sdpa",
            device_map=device_map
        )
        self.text_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, text_forward_outs, input_ids, attention_mask):
        text_features = text_forward_outs.hidden_states[self.select_layer]
        
        if self.text_tower.text_model.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = text_features[
                torch.arange(text_features.shape[0], device=text_features.device),
                input_ids.to(dtype=torch.int, device=text_features.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = text_features[
                torch.arange(text_features.shape[0], device=text_features.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=text_features.device) == self.text_tower.text_model.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]
        
        return text_features, attention_mask

    def forward(self, texts):
        if type(texts) is list:
            text_features = []
            for text in texts:
                text_forward_out = self.text_tower(**text, output_hidden_states=True)
                text_feature = self.feature_select(text_forward_out, text)
                text_features.append(text_feature)
        else:
            text_forward_out = self.text_tower(**texts, output_hidden_states=True)
            text_features = self.feature_select(text_forward_out, **texts)
            
        return text_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.text_tower.dtype

    @property
    def device(self):
        return self.text_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.text_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        # if "slicefour" in self.select_feature:
        #     _hidden_size *= 4
        # if "slice_m25811_f6" in self.select_feature:
        #     _hidden_size *= 5
        return _hidden_size


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        if isinstance(self.select_layer, str):
            self.select_layer = self.select_layer.split(',')
            self.select_layer = [int(layer) for layer in self.select_layer]
            if len(self.select_layer) == 1:
                self.select_layer = self.select_layer[0]
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            # TODO: better detector is needed.
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(args, "mm_tunable_parts") and "mm_vision_tower" in args.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map="cuda"):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor = CLIPImageProcessorFast.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name, 
            attn_implementation="flash_attention_2",
            device_map=device_map
        )
        self.vision_tower.requires_grad_(False)
        # self.vision_tower.vision_model.embeddings.register_forward_hook(self._remove_cls_token)
        # self.vision_tower.vision_model.embeddings.register_forward_hook(self._modify_text_features)
        # self.vision_tower.vision_model.encoder.register_forward_pre_hook(self._add_attention_mask, with_kwargs=True)
        self.text_features = None
        self.attention_mask = None
        self.img_seq_len = (self.vision_tower.config.image_size // self.vision_tower.config.patch_size) ** 2

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        select_feature_type = self.select_feature

        if self.select_feature in ["slicefour_patch", "slicefour_cls_patch"]:
            select_every_k_layer = len(image_forward_outs.hidden_states) // 4
            image_features = torch.stack([image_forward_outs.hidden_states[i] for i in range(select_every_k_layer + self.select_layer, len(image_forward_outs.hidden_states), select_every_k_layer)], dim=0)
            select_feature_type = select_feature_type.replace("slicefour_", "")
        elif self.select_feature in ["slice_m25811_f6_patch", "slice_m25811_f6_cls_patch"]:
            select_layers = [-2, -5, -8, -11, 6]
            image_features = torch.stack([image_forward_outs.hidden_states[i] for i in select_layers], dim=0)
            select_feature_type = select_feature_type.replace("slice_m25811_f6_", "")
        elif self.select_feature in ["slice_f2610141822_patch", "slice_f2610141822_cls_patch"]:
            select_layers = [22, 18, 14, 10, 6, 2]
            image_features = torch.stack([image_forward_outs.hidden_states[i] for i in select_layers], dim=0)
            select_feature_type = select_feature_type.replace("slice_f2610141822_", "")
        elif self.select_feature.startswith("slice_last_") and ("_patch" in self.select_feature or "_cls_patch" in self.select_feature):
            num_layers = int(self.select_feature.split("_")[2])
            select_layers = list(range(-2, -2-num_layers, -1))
            image_features = torch.stack([image_forward_outs.hidden_states[i] for i in select_layers], dim=0)
            select_feature_type = select_feature_type.replace(f"slice_last_{num_layers}_", "")
        else:
            if isinstance(self.select_layer, list):
                image_features = torch.stack([image_forward_outs.hidden_states[i] for i in self.select_layer], dim=0)
            else:
                image_features = image_forward_outs.hidden_states[self.select_layer]
        
        if select_feature_type == "patch":
            image_features = image_features[..., 1:self.img_seq_len+1, :]
        elif select_feature_type == "cls_patch":
            image_features = image_features[..., 0:self.img_seq_len+1, :]
        else:
            raise ValueError(f"Unexpected select feature: {select_feature_type}")
        return image_features
    
    def get_last_attention_scores(self, image_forward_outs):
        last_layer = self.vision_tower.vision_model.encoder.layers[-1]
        last_attn = last_layer.self_attn
        hidden_states = image_forward_outs.hidden_states[-2]
        hidden_states = last_layer.layer_norm1(hidden_states)
        
        bsz, tgt_len, embed_dim = hidden_states.size()
        
        if self.select_feature == "patch":
            image_states = hidden_states[:, 1:self.img_seq_len+1]
        elif self.select_feature == "cls_patch":
            image_states = hidden_states[:, :self.img_seq_len+1]
        text_states = hidden_states[:, self.img_seq_len+1:]
        
        image_len = image_states.size(1)
        text_len = text_states.size(1)
        dtype = hidden_states.dtype

        # get query proj
        query_states = last_attn.q_proj(text_states) * last_attn.scale
        key_states = last_attn._shape(last_attn.k_proj(image_states), -1, bsz)

        proj_shape = (bsz * last_attn.num_heads, -1, last_attn.head_dim)
        query_states = last_attn._shape(query_states, text_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        
        attention_mask = self.attention_mask[:, None, :, None].expand(bsz, 1, text_len, image_len).to(dtype)
        attention_mask = 1.0 - attention_mask
        attention_mask = attention_mask.masked_fill(attention_mask.to(torch.bool), torch.finfo(dtype).min)
        
        attn_weights = attn_weights.view(bsz, last_attn.num_heads, text_len, image_len) + attention_mask
        
        text_scores = nn.functional.softmax(attn_weights, dim=-2).mean(-1)  # [bsz, num_heads, text_len]
        scores = text_scores.unsqueeze(-1) * nn.functional.softmax(attn_weights, dim=-1)  # [bsz, num_heads, text_len, image_len]
        image_scores = scores.mean(1).sum(-2) * image_len  # [bsz, image_len]
        
        # topk_values, topk_indices = torch.topk(image_scores, k=int(0.5*image_len), dim=-1)
        # mask = torch.zeros_like(image_scores)
        # mask.scatter_(-1, topk_indices, 1)
        
        return image_scores
        

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # last_image_features = image_forward_outs.pooled_output
            # if getattr(image_forward_outs, "attentions", None) is not None:
            #     for attention in image_forward_outs.attentions:
            #         attention.retain_grad()
            #     self.output_attention = image_forward_outs.attentions
                
        # if self.attention_mask is not None:
        #     image_scores = self.get_last_attention_scores(image_forward_outs)
        #     image_features = image_scores.unsqueeze(-1) * image_features
                
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        _hidden_size = self.config.hidden_size
        # if "slicefour" in self.select_feature:
        #     _hidden_size *= 4
        # if "slice_m25811_f6" in self.select_feature:
        #     _hidden_size *= 5
        return _hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        _num_patches = (self.config.image_size // self.config.patch_size) ** 2
        if "cls_patch" in self.select_feature:
            _num_patches += 1
        return _num_patches

    @property
    def image_size(self):
        return self.config.image_size

    # def _remove_cls_token(self, module, input, output):
    #     output = output[:, 1:]
    #     return output
    
    def _modify_text_features(self, module, input, output):
        if self.text_features is not None:
            # output[:, 0] = self.text_features + module.position_embedding(module.position_ids[:, 0])
            assert self.text_features is not None and getattr(self, "text_pos_embedding", None) is not None
            output = torch.concat([
                output, 
                self.text_features + self.text_pos_embedding(module.position_ids[:, 0:self.text_features.shape[1]])
            ], dim=1)
            
            return output
        
    def _add_attention_mask(self, module, inputs, kwargs):
        if self.attention_mask is not None:
            batch_size = self.attention_mask.size(0)
            img_seq_len = kwargs['inputs_embeds'].size(1) - self.attention_mask.size(1)
            kwargs['attention_mask'] = torch.cat([
                torch.ones((batch_size, img_seq_len), dtype=self.attention_mask.dtype, device=self.attention_mask.device),
                self.attention_mask
            ], dim=1)
            if self.vision_tower.config._attn_implementation != "flash_attention_2":
                kwargs['attention_mask'] = _prepare_4d_attention_mask(kwargs['attention_mask'], kwargs['inputs_embeds'].dtype)
        
            return inputs, kwargs
        
    
    def modify_text_features(self, text_features, attention_mask=None):
        self.text_features = text_features
        self.attention_mask = attention_mask


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):

        self.s2_scales = getattr(args, "s2_scales", "336,672,1008")
        self.s2_scales = list(map(int, self.s2_scales.split(",")))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        super().__init__(vision_tower, args, delay_load)

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size["shortest_edge"] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

        self.is_loaded = True

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)
                image_features.append(image_feature)
        else:
            image_features = multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size, split_forward=True)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)

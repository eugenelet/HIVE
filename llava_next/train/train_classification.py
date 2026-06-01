import os
import pathlib
import torch
import numpy as np
import transformers
from PIL import Image
import warnings

from torch import nn
from torchvision import transforms
from timm.data.auto_augment import rand_augment_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from transformers import AutoModelForImageClassification, AutoImageProcessor, Trainer, DefaultDataCollator
from transformers.models.siglip.modeling_siglip import SiglipConfig, SiglipForImageClassification, SiglipVisionTransformer
from transformers.models.clip.modeling_clip import CLIPConfig, CLIPPreTrainedModel, CLIPVisionTransformer, ImageClassifierOutput, CLIPForImageClassification
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.siglip.modeling_siglip import lecun_normal_
import datasets
from typing import Optional
from dataclasses import dataclass, field

from llava_next.model.builder import load_pretrained_model
from llava_next.mm_utils import get_model_name_from_path
from llava_next.utils import rank0_print

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_base: Optional[str] = field(default=None)
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    text_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_text_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_text_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)
    mm_cross_select_layer: Optional[str] = field(default=None)
    pretrain_other_parameters: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

    mm_vision_tower_parameters: Optional[str] = field(default=None)

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default="cifar100")
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    cross_enable: bool = field(default=False)
    flamingo_enable: bool = field(default=False)
    prompt_aware_enable: bool = field(default=False)
    lit_enable: bool = field(default=False)
    tune_vision_enable: bool = field(default=False)

def compute_metrics(pred):
    logits, labels = pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    return {"accuracy": accuracy}

# Define Lazy Preprocessing
class LazyPreprocessor:
    def __init__(self, image_processor, dataset_name, transform=transforms.Compose([])):
        self.image_processor = image_processor
        self.transform = transform
        self.dataset_name = dataset_name

    def __call__(self, examples):
        try:
            pixel_values = [
                self.image_processor(self.transform(
                    image.convert("RGB")
                ))["pixel_values"][0] for image in examples["img"]
            ]
        except:
            pixel_values = [
                self.image_processor(self.transform(
                    image.convert("RGB")
                ))["pixel_values"][0] for image in examples["image"]
            ]

        if "Caltech-256" in self.dataset_name:
            labels = [label - 1 for label in examples["label"]]
        else:
            labels = examples["label"]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

    
class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return torch.zeros_like(x)
    
class AttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, config):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state, need_weights=False)[0]

        return hidden_state[:, 0]
    
class ImageClassificationAttentiveProbe(nn.Module):
    def __init__(self, vision_model) -> None:
        super().__init__()

        self.config = vision_model.config
        self.num_labels = vision_model.num_labels
        self.vision_model = vision_model.vision_model

        self.post_layernorm = nn.LayerNorm(self.config.vision_config.hidden_size, eps=self.config.vision_config.layer_norm_eps)
        self.post_layernorm.bias.data.zero_()
        self.post_layernorm.weight.data.fill_(1.0)
        self.head = AttentionPoolingHead(self.config.vision_config)
        nn.init.xavier_uniform_(self.head.probe.data)
        nn.init.xavier_uniform_(self.head.attention.in_proj_weight.data)
        nn.init.zeros_(self.head.attention.in_proj_bias.data)
        lecun_normal_(self.head.attention.out_proj.weight)
        nn.init.zeros_(self.head.attention.out_proj.bias)

        # Classifier head
        self.classifier = vision_model.classifier

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if isinstance(self.vision_model, CLIPVisionTransformer):
            # CLIP need to remove the CLS token
            sequence_output = sequence_output[:, 1:, :]

        # Attentive probe
        sequence_output = self.head(self.post_layernorm(sequence_output))

        # apply classifier
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    image_processor = AutoImageProcessor.from_pretrained(model_args.vision_tower, cache_dir=training_args.cache_dir, use_fast=True)
    
    dataset = datasets.load_dataset(data_args.dataset_name, cache_dir=os.path.join(data_args.data_path, data_args.dataset_name))
    
    if "label" not in dataset["train"].features:  # for cifar100
        dataset = dataset.rename_column("fine_label", "label")
        dataset = dataset.remove_columns("coarse_label")
    if "cifar100" in data_args.dataset_name:
        num_labels = 100
    elif "cifar10" in data_args.dataset_name:
        num_labels = 10
    elif "tiny-imagenet" in data_args.dataset_name:
        num_labels = 200
    elif "Caltech-256" in data_args.dataset_name:
        num_labels = 257
    elif "imagenet-1k" in data_args.dataset_name:
        num_labels = 1000
    elif "food101" in data_args.dataset_name:
        num_labels = 101
    elif "stanford_cars" in data_args.dataset_name:
        num_labels = 196
    elif "oxford-iiit-pet" in data_args.dataset_name:
        num_labels = 37
    # num_labels = len(dataset["train"].features["label"].names)

    # Transform only for training
    transform = transforms.Compose([
        transforms.RandomResizedCrop(
            size=list(image_processor.size.values())[0],
            scale=(0.4, 1),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)
    ])

    # Define lazy preprocessors for train and eval splits
    lazy_train_preprocessor = LazyPreprocessor(image_processor, data_args.dataset_name, transform)
    lazy_eval_preprocessor = LazyPreprocessor(image_processor, data_args.dataset_name)  # No transformation for evaluation

    # Apply preprocessing to the respective splits
    train_dataset = dataset["train"].with_transform(lazy_train_preprocessor)
    try:
        eval_dataset = dataset["validation"].with_transform(lazy_eval_preprocessor)
    except:
        try:
            eval_dataset = dataset["valid"].with_transform(lazy_eval_preprocessor)
        except:
            eval_dataset = dataset["test"].with_transform(lazy_eval_preprocessor)
    data_collator = DefaultDataCollator()

    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    model = AutoModelForImageClassification.from_pretrained(
        model_args.vision_tower,
        attn_implementation=training_args.attn_implementation,
        cache_dir=training_args.cache_dir,
        torch_dtype=compute_dtype,
        num_labels=num_labels, 
        device_map="cpu"
    )

    if model_args.mm_vision_tower_parameters is not None:
        rank0_print(f"Loading vision tower parameters from {model_args.mm_vision_tower_parameters}")
        vision_tower_weight_dict = torch.load(model_args.mm_vision_tower_parameters, map_location="cpu", weights_only=True)
        vision_tower_weight_dict = {(k[32:] if k.startswith("model.vision_tower.vision_tower.") else k): v for k, v in vision_tower_weight_dict.items()}
        model.load_state_dict(vision_tower_weight_dict, strict=False)
    
    # Attentive probe
    del model.vision_model.encoder.layers[-1:]
    model.vision_model.post_layernorm = nn.Identity()
    model = ImageClassificationAttentiveProbe(model)
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    model.post_layernorm.requires_grad_(True)
    model.classifier.requires_grad_(True)

    # Linear probe
    # del model.vision_model.encoder.layers[-1:]
    # model.vision_model.post_layernorm.bias.data.zero_()
    # model.vision_model.post_layernorm.weight.data.fill_(1.0)
    # model.requires_grad_(False)
    # model.vision_model.post_layernorm.requires_grad_(True)
    # model.classifier.requires_grad_(True)

    rank0_print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name, param.shape)

    if training_args.lr_scheduler_type == "cosine_with_min_lr":
        training_args.lr_scheduler_kwargs = {
            'min_lr_rate': 0.1
        }

    warnings.filterwarnings("ignore", module="PIL.TiffImagePlugin")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=image_processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

if "__main__" in __name__:
    train()
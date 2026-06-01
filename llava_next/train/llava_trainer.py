import os
import torch
import torch.nn as nn
import datetime

from torch.utils.data import Dataset, Sampler, DataLoader

from trl.trainer import DPOTrainer

from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, ALL_LAYERNORM_LAYERS, logger, is_accelerate_available, is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from typing import List, Optional
from datetime import timedelta
from accelerate.utils import DistributedType

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches, InitProcessGroupKwargs

if is_datasets_available():
    import datasets

from llava_next.utils import rank0_print

if is_accelerate_available("0.28.0"):
    from accelerate.utils import DataLoaderConfiguration


from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.training_args import OptimizerNames


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


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    # def create_accelerator_and_postprocess(self):
    #     grad_acc_kwargs = {"num_steps": self.args.gradient_accumulation_steps}
    #     grad_acc_kwargs["sync_with_dataloader"] = False
    #     gradient_accumulation_plugin = GradientAccumulationPlugin(**grad_acc_kwargs)

    #     accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
    #     rank0_print("Setting NCCL timeout to INF to avoid running errors.")

    #     # create accelerator object
    #     self.accelerator = Accelerator(
    #         dispatch_batches=self.args.dispatch_batches, split_batches=self.args.split_batches, deepspeed_plugin=self.args.deepspeed_plugin, gradient_accumulation_plugin=gradient_accumulation_plugin, kwargs_handlers=[accelerator_kwargs]
    #     )
    #     # some Trainer classes need to use `gather` instead of `gather_for_metrics`, thus we store a flag
    #     self.gather_function = self.accelerator.gather_for_metrics

    #     # deepspeed and accelerate flags covering both trainer args and accelerate launcher
    #     self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
    #     self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

    #     # post accelerator creation setup
    #     if self.is_fsdp_enabled:
    #         fsdp_plugin = self.accelerator.state.fsdp_plugin
    #         fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", fsdp_plugin.limit_all_gathers)
    #         if is_accelerate_available("0.23.0"):
    #             fsdp_plugin.activation_checkpointing = self.args.fsdp_config.get("activation_checkpointing", fsdp_plugin.activation_checkpointing)
    #             if fsdp_plugin.activation_checkpointing and self.args.gradient_checkpointing:
    #                 raise ValueError("The activation_checkpointing in FSDP config and the gradient_checkpointing in training arg " "can't be set to True simultaneously. Please use FSDP's activation_checkpointing logic " "when using FSDP.")

    #     if self.is_deepspeed_enabled and getattr(self.args, "hf_deepspeed_config", None) is None:
    #         self.propagate_args_to_deepspeed()
    
    # def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
    #     """
    #     Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
    #     handling potential state.
    #     """
    #     inputs = self._prepare_input(inputs)
    #     if len(inputs) == 0:
    #         raise ValueError(
    #             "The batch received was empty, your model won't be able to train on it. Double-check that your "
    #             f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
    #         )
    #     if self.args.past_index >= 0 and self._past is not None:
    #         inputs["mems"] = self._past
            
    #     inputs["gradient_accumulation_steps"] = self.args.gradient_accumulation_steps
        
    #     return inputs
    
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # if not self.args.alt_enable:
        #     return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        
        # del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # Finally we need to normalize the loss for reporting
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            loss = loss / self.args.gradient_accumulation_steps

        # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            kwargs["scale_wrt_gas"] = False

        self.accelerator.backward(loss, **kwargs)

        # breakpoint()

        import torch
        import numpy as np
        import os
        import matplotlib.pyplot as plt
        from PIL import Image

        # Empty the folder
        for file in os.listdir("image"):
            os.remove(os.path.join("image", file))

        # Define normalization parameters (assuming ImageNet normalization)
        IMAGENET_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        IMAGENET_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

        def tensor_to_image(tensor):
            """Convert a normalized tensor back to an image (H, W, C)."""
            if tensor.dim() == 4:
                tensor = tensor[0]  # Remove batch dimension if present

            tensor = tensor * IMAGENET_STD.to(tensor.device) + IMAGENET_MEAN.to(tensor.device)  # De-normalize
            tensor = tensor.clamp(0, 1)  # Ensure values are in valid range
            image = tensor.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
            return (image * 255).astype(np.uint8)  # Convert to uint8

        def overlay_heatmap_on_image(image, heatmap, alpha=0.7):
            """Overlay a heatmap on the original image using PIL and matplotlib colormap."""
            heatmap = np.array(Image.fromarray(heatmap)) # .resize((image.shape[1], image.shape[0]), Image.Resampling.BICUBIC))

            cmap = plt.get_cmap('jet')
            heatmap_colored = cmap(heatmap)[:, :, :3]  # Drop alpha channel
            heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

            return heatmap_colored
            # overlay = (image * (1 - alpha) + heatmap_colored * alpha).astype(np.uint8)
            # return overlay

        def reshape_patch_embedding(grad_map, grid_size=(24, 24)):
            return grad_map.reshape(grid_size)

        def save_heatmap(image, heatmap, save_path):
            """Save the heatmap overlay using PIL."""
            heatmap_image = overlay_heatmap_on_image(image, heatmap)
            Image.fromarray(heatmap_image).save(save_path)

        # Get the original image from input
        original_image = tensor_to_image(inputs["images"][0])

        # Get gradients from the vision tower
        hidden_states = model.get_vision_tower().hidden_states  # List of layer outputs
        print(f"Number of Vision Encoder Layers: {len(hidden_states)}")

        # # Collect all gradients to compute global min and max
        # all_grads = []

        # for layer_idx, layer_output in enumerate(hidden_states):
        #     if layer_output.grad is None:
        #         print(f"Layer {layer_idx}: No gradient found, skipping.")
        #         continue

        #     # Average gradients over embedding dimension
        #     grad_map = layer_output.grad.mean(dim=-1).squeeze().float().cpu().numpy()  # Shape: [num_patches]
        #     all_grads.append(grad_map)

        # # Compute global min and max for normalization
        # all_grads = np.concatenate([g.flatten() for g in all_grads])  # Flatten and concatenate all layers
        # global_min, global_max = all_grads.min(), all_grads.max()

        def normalize_heatmap_global(grad_map, max_value, min_value):
            """Normalize a heatmap using global min and max."""
            return (grad_map - min_value) / (max_value - min_value + 1e-8)  # Normalize with global range

        # Process each layer with global normalization
        for layer_idx, layer_output in enumerate(hidden_states):
            grad = layer_output.grad  # [1, num_tokens, dim]
            if grad is None:
                print(f"Layer {layer_idx} has no gradients.")
                continue

            # Focus only on patch tokens
            patch_grads = grad[0]           # [num_patches, dim]

            # Compute L2-norm of gradient for each patch
            grad_map = patch_grads.norm(dim=1)  # [num_patches]

            grad_map = grad_map.detach().float().cpu().numpy()
            grad_map = reshape_patch_embedding(grad_map)  # e.g., (24, 24)

            # Normalize the heatmap
            max_grad = grad_map.max()
            min_grad = grad_map.min()
            grad_map = normalize_heatmap_global(grad_map, max_grad, min_grad)  # Normalize using global min and max

            # Save heatmap image
            save_path = os.path.join("image", f"vision_layer_{layer_idx:02d}_grad.png")
            save_heatmap(original_image, grad_map, save_path)
            print(f"Saved Layer {layer_idx} gradient heatmap to {save_path}")

        # Save original image
        save_path = os.path.join("image", f"original.png")
        Image.fromarray(original_image).save(save_path)

        # breakpoint()


        def create_grid_image(image_dir, save_path, grid_size=(5, 5), image_size=(384, 384), gap_size=10, background_color=(255, 255, 255)):
            """
            Create a 5x5 grid image with the original image at the top-left corner and gaps between images.

            Args:
                image_dir (str): Directory containing the saved heatmap images.
                save_path (str): Path to save the final grid image.
                grid_size (tuple): Number of rows and columns in the grid (default: 5x5).
                image_size (tuple): Size of each individual image (default: 384x384).
                gap_size (int): Gap size (in pixels) between images.
                background_color (tuple): RGB color for the gap background (default: white).
            """
            # Get all heatmap image paths (excluding the original image)
            heatmap_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png") and "original" not in f])

            # Load original image
            original_path = os.path.join(image_dir, "original.png")
            if not os.path.exists(original_path):
                print("Error: Original image not found!")
                return

            original_image = Image.open(original_path).resize(image_size, Image.Resampling.BICUBIC)

            # Check if we have enough heatmaps
            num_images = grid_size[0] * grid_size[1] - 1  # Reserve 1 spot for the original image
            if len(heatmap_paths) < num_images:
                print(f"Warning: Expected {num_images} heatmaps, but found {len(heatmap_paths)}.")

            # Load heatmap images and resize
            heatmaps = [Image.open(path).resize(image_size, Image.Resampling.NEAREST) for path in heatmap_paths[:num_images]]

            # Calculate final canvas size (including gaps)
            total_width = grid_size[1] * image_size[0] + (grid_size[1] - 1) * gap_size
            total_height = grid_size[0] * image_size[1] + (grid_size[0] - 1) * gap_size

            # Create blank canvas with the background color
            grid_image = Image.new("RGB", (total_width, total_height), background_color)

            # Place the original image at the top-left (0,0), considering gaps
            grid_image.paste(original_image, (0, 0))

            # Paste heatmaps into the grid (starting from index 1)
            for idx, img in enumerate(heatmaps):
                row, col = divmod(idx + 1, grid_size[1])  # Start from index 1 to leave (0,0) for the original image
                x_offset = col * (image_size[0] + gap_size)
                y_offset = row * (image_size[1] + gap_size)
                grid_image.paste(img, (x_offset, y_offset))

            # Save the final grid image
            save_path = os.path.join(image_dir, save_path)
            grid_image.save(save_path)
            print(f"Grid image with gaps saved at: {save_path}")

        # # Directory where the heatmaps and original image are stored
        image_dir = "image"
        save_path = "gradient_heatmap_grid.png"

        # Create the 5x5 grid image with gaps
        create_grid_image(image_dir, save_path, gap_size=10, background_color=(255, 255, 255))  # White background, 15px gaps

        print(model.tokenizer.decode(inputs["input_ids"][0][inputs["input_ids"][0].ne(-200)]))
        tokens = inputs["input_ids"][0][inputs["input_ids"][0].ne(-200)]
        for i, cross_block in enumerate(model.target_modules):
            attn_weight = cross_block.attn_weights[0][-1, :, 1:577]

            for j, token in enumerate(tokens):
                text = model.tokenizer.decode(token)

                attention_map = attn_weight[j].reshape(24, 24).detach().float().cpu().numpy()
                attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
                save_path = os.path.join("image", f"cross_layer{i}_token{j}_{text}.png")

                save_heatmap(original_image, attention_map, save_path)

        # breakpoint()


        def plot_filtered_attention_maps(image_dir, save_path, num_layers, tokens, selected_tokens):
            """
            Create a visualization of cross-attention maps for only the selected tokens.

            Args:
                image_dir (str): Directory where attention maps are stored.
                save_path (str): Path to save the final figure.
                num_layers (int): Number of cross-attention layers.
                tokens (list): List of all token texts.
                selected_tokens (list): Tokens to be included in the visualization.
            """
            # Load the original image
            original_path = os.path.join(image_dir, "original.png")
            if not os.path.exists(original_path):
                print("Error: Original image not found!")
                return

            original_image = Image.open(original_path)
            
            # Filter tokens to only keep selected ones
            if selected_tokens is not None:
                filtered_indices = [i for i, t in enumerate(tokens) if t in selected_tokens]
            else:
                filtered_indices = range(len(tokens))
            filtered_tokens = [tokens[i] for i in filtered_indices]
            num_tokens = len(filtered_tokens)

            if num_tokens == 0:
                print("Error: None of the selected tokens were found in the input!")
                return

            # Create figure with reduced gaps
            fig, axes = plt.subplots(num_layers, num_tokens, figsize=(num_tokens * 1.5, num_layers * 1.5))

            # Ensure axes is a list in case there's only one row or column
            if num_layers == 1:
                axes = [axes]
            if num_tokens == 1:
                axes = [[ax] for ax in axes]

            for i in range(num_layers):
                for j, token_idx in enumerate(filtered_indices):
                    token_text = tokens[token_idx]

                    # Load the attention heatmap
                    heatmap_path = os.path.join(image_dir, f"cross_layer{i}_token{token_idx}_{token_text}.png")
                    if not os.path.exists(heatmap_path):
                        print(f"Warning: Missing heatmap for Layer {i}, Token {token_idx} ({token_text})")
                        continue
                    
                    heatmap = Image.open(heatmap_path)

                    # Plot the heatmap
                    ax = axes[i][j]  # Access the correct subplot
                    ax.imshow(heatmap)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_frame_on(False)

            # Adjust layout to make space for labels
            plt.subplots_adjust(left=0.15, bottom=0.2, right=0.98, top=0.95, wspace=0.02, hspace=0.02)

            # Add X-axis labels (token text) at the middle of columns
            for j in range(num_tokens):
                axes[-1][j].set_xlabel(filtered_tokens[j], fontsize=12, fontweight="bold", labelpad=6)

            # Add Y-axis labels (layer numbers) at the middle of rows
            for i in range(num_layers):
                axes[i][0].set_ylabel(f"Layer {i}", fontsize=14, fontweight="bold", rotation=90, labelpad=10, va="center")

            # Save final figure
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            print(f"Filtered attention map grid saved at: {save_path}")

        # Directory where the attention maps and original image are stored
        image_dir = "image"
        save_path = os.path.join(image_dir, "filtered_attention_map_grid.png")

        # Extract token texts
        tokens = [model.tokenizer.decode(token) for token in inputs["input_ids"][0][inputs["input_ids"][0].ne(-200)].tolist()]
        num_layers = len(model.target_modules)

        # Define the specific tokens to keep
        # selected_tokens = ["a", "words", "today", "cross", "logo"]

        # Create the filtered attention visualization
        plot_filtered_attention_maps(image_dir, save_path, num_layers, tokens, None)

        breakpoint()

        # print(model.tokenizer.decode(inputs["input_ids"][0][inputs["input_ids"][0].ne(-200)]))
        # tokens = inputs["input_ids"][0][inputs["input_ids"][0].ne(-200)]
        # for i, cross_block in enumerate(model.target_modules):
        #     attn_weight = cross_block.attn_weights[0][-1, :, :]

        #     with open(os.path.join("image", f"cross_layer{i}.txt"), "w") as f:
        #         f.write(f"Tokens: {model.tokenizer.decode(inputs['input_ids'][0][inputs['input_ids'][0].ne(-200)])}\n\n")
        #         for j, token in enumerate(tokens):
        #             text = model.tokenizer.decode(token)
        #             attention_map = attn_weight[j].detach().float().cpu().numpy()
        #             image_attn = attention_map[1:577].sum()
        #             text_attn = attention_map[577:].sum()

        #             f.write(f"Token: {text}, Image Attention: {image_attn}, Text Attention: {text_attn}\n")
        
        # breakpoint()

        return loss.detach()

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
            )
        elif self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality=True,
            )
        elif self.args.group_by_modality_length_auto:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                group_by_modality_auto=True,
            )
        elif self.args.group_by_varlen:
            lengths = self.train_dataset.lengths
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                # self.args.train_batch_size, # TODO: seems that we should have gradient_accumulation_steps
                # world_size=self.args.world_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,  # TODO: seems that this may work?
                lengths=lengths,
                variable_length=True,
            )
        else:
            return super()._get_train_sampler()

    # def get_train_dataloader(self) -> DataLoader:
    #     """
    #     Returns the training [`~torch.utils.data.DataLoader`].

    #     Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    #     training if necessary) otherwise.

    #     Subclass and override this method if you want to inject some custom behavior.
    #     """
    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")

    #     train_dataset = self.train_dataset
    #     data_collator = self.data_collator
    #     if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
    #         train_dataset = self._remove_unused_columns(train_dataset, description="training")
    #     else:
    #         data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    #     dataloader_params = {
    #         "batch_size": self._train_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #         "persistent_workers": self.args.dataloader_persistent_workers,
    #     }

    #     if not isinstance(train_dataset, torch.utils.data.IterableDataset):
    #         dataloader_params["sampler"] = self._get_train_sampler()
    #         dataloader_params["drop_last"] = self.args.dataloader_drop_last
    #         dataloader_params["worker_init_fn"] = seed_worker
    #         dataloader_params["prefetch_factor"] = self.args.dataloader_num_workers * 2 if self.args.dataloader_num_workers != 0 else None

    #     dataloader = self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    #     return dataloader

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            lr_mapper = {}
            wd_mapper = {}
            
            if self.args.mm_projector_lr is not None:
                # lr_mapper["mm_projector"] = self.args.mm_projector_lr 
                # lr_mapper["mm_text_projector"] = self.args.mm_projector_lr
                # lr_mapper["multimodal"] = self.args.mm_projector_lr
                lr_mapper['mm_proj'] = self.args.mm_projector_lr
                lr_mapper['q_norm'] = self.args.mm_projector_lr
                lr_mapper['k_norm'] = self.args.mm_projector_lr
            
            if self.args.mm_vision_tower_lr is not None:
                lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            
            if self.args.mm_projector_wd is not None:
                wd_mapper["mm_projector"] = self.args.mm_projector_wd 
                # wd_mapper["mm_text_projector"] = self.args.mm_projector_wd
                # wd_mapper["multimodal"] = self.args.mm_projector_wd
                wd_mapper['mm_proj'] = self.args.mm_projector_wd
            
            if self.args.mm_vision_tower_wd is not None:
                wd_mapper["vision_tower"] = self.args.mm_vision_tower_wd
            
            if len(lr_mapper) > 0 or len(wd_mapper) > 0:
                special_parameters = set(lr_mapper.keys()).union(wd_mapper.keys())
                grouped_params = {}
                
                def add_to_group(params, lr, wd):
                    key = (lr, wd)
                    if key not in grouped_params:
                        grouped_params[key] = []
                    grouped_params[key].extend(params)
                
                # Standard parameters (not in special modules)
                standard_decay_params = [
                    p for n, p in opt_model.named_parameters() if n in decay_parameters and p.requires_grad and not any(k in n for k in special_parameters)
                ]
                add_to_group(standard_decay_params, None, self.args.weight_decay)
                
                standard_non_decay_params = [
                    p for n, p in opt_model.named_parameters() if n not in decay_parameters and p.requires_grad and not any(k in n for k in special_parameters)
                ]
                add_to_group(standard_non_decay_params, None, 0.0)
                
                # Special Parameter Groups
                for module_keyword in special_parameters:
                    module_parameters_decay = [
                        p for n, p in opt_model.named_parameters() if n in decay_parameters and module_keyword in n and p.requires_grad
                    ]
                    module_parameters_no_decay = [
                        p for n, p in opt_model.named_parameters() if n not in decay_parameters and module_keyword in n and p.requires_grad
                    ]
                    
                    weight_decay_value = wd_mapper.get(module_keyword, self.args.weight_decay)
                    lr_value = lr_mapper.get(module_keyword, None)
                    
                    add_to_group(module_parameters_decay, lr_value, weight_decay_value)
                    add_to_group(module_parameters_no_decay, lr_value, 0.0)
                
                optimizer_grouped_parameters = []
                for (lr, wd), params in grouped_params.items():
                    if params:
                        if lr is None:
                            optimizer_grouped_parameters.append({"params": params, "weight_decay": wd})
                        else:
                            optimizer_grouped_parameters.append({"params": params, "lr": lr, "weight_decay": wd})

            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]

            # lr_mapper = {}
            # if self.args.mm_projector_lr is not None:
            #     # lr_mapper["mm_projector"] = self.args.mm_projector_lr
            #     lr_mapper["mm_text_projector"] = self.args.mm_projector_lr
            #     lr_mapper["multimodal"] = self.args.mm_projector_lr
            #     lr_mapper['mm_proj'] = self.args.mm_projector_lr
            #     lr_mapper['q_norm'] = self.args.mm_projector_lr
            #     lr_mapper['k_norm'] = self.args.mm_projector_lr
            # if self.args.mm_vision_tower_lr is not None:
            #     # Apply Layer-wise Learning Rate Decay (LLRD)
            #     # vision_tower.vision_model.encoder.layers.0~25
            #     # for i in range(26):
            #     #     name = f"vision_tower.vision_model.encoder.layers.{i}."
            #     #     lr_mapper[name] = self.args.mm_vision_tower_lr * (0.9 ** (25 - i))
            #     #     rank0_print(f"Vision Layer {i} LR: {lr_mapper[name]}")
            #     lr_mapper["vision_tower"] = self.args.mm_vision_tower_lr
            #     # print(lr_mapper)
            # if len(lr_mapper) > 0:
            #     special_lr_parameters = [name for name, _ in opt_model.named_parameters() if any(module_keyword in name for module_keyword in lr_mapper)]
            #     optimizer_grouped_parameters = [
            #         {
            #             "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
            #             "weight_decay": self.args.weight_decay,
            #         },
            #         {
            #             "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
            #             "weight_decay": 0.0,
            #         },
            #     ]
            #     for module_keyword, lr in lr_mapper.items():
            #         module_parameters = [name for name, _ in opt_model.named_parameters() if module_keyword in name]
            #         optimizer_grouped_parameters.extend(
            #             [
            #                 {
            #                     "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in module_parameters and p.requires_grad)],
            #                     "weight_decay": self.args.weight_decay,
            #                     "lr": lr,
            #                 },
            #                 {
            #                     "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in module_parameters and p.requires_grad)],
            #                     "weight_decay": 0.0,
            #                     "lr": lr,
            #                 },
            #             ]
            #         )
            # else:
            #     optimizer_grouped_parameters = [
            #         {
            #             "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
            #             "weight_decay": self.args.weight_decay,
            #         },
            #         {
            #             "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
            #             "weight_decay": 0.0,
            #         },
            #     ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs["fused"] = True

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        if (getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) >= 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        )) and not self.args.lora_enable:
            # Save Adapter logic
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            
            other_key_to_match = []
            for n, p in self.model.named_parameters():
                if p.requires_grad and not any(key_match in n for key_match in keys_to_match):
                    other_key_to_match.append(n)
            other_weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), other_key_to_match)

            if hasattr(self.args, "mm_tunable_parts") and "mm_vision_tower" in self.args.mm_tunable_parts:
                vision_key_to_match = ["vision_tower"]
                vision_weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), vision_key_to_match)
            else:
                vision_weight_to_save = {}

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
                torch.save(other_weight_to_save, os.path.join(output_dir, "other_parameters.bin"))
                torch.save(vision_weight_to_save, os.path.join(output_dir, "vision_tower.bin"))

        elif self.args.lora_enable:
            from transformers.modeling_utils import unwrap_model
            unwrapped_model = unwrap_model(model)

            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(unwrapped_model.named_parameters())
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                if hasattr(unwrapped_model, "config"):
                    unwrapped_model.config.save_pretrained(output_dir)
                if hasattr(unwrapped_model, "generation_config"):
                    unwrapped_model.generation_config.save_pretrained(output_dir)
                torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_trainables.bin"))

        super(LLaVATrainer, self)._save_checkpoint(model, trial)

        # Ensure only rank 0 creates or updates the symlink
        if self.args.local_rank == 0 or self.args.local_rank == -1:
            symlink_path = os.path.join(self.args.output_dir, "checkpoint-last")

            # First, check if it exists before attempting to remove
            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                try:
                    os.unlink(symlink_path)  # Remove the old symlink or file
                except FileNotFoundError:
                    pass  # Ignore if file was already deleted in another process

            # Ensure only rank 0 creates the new symlink
            if not os.path.exists(symlink_path):  
                os.symlink(checkpoint_folder, symlink_path)  # Create a new symlink


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
            

class LLaVADPOTrainer(DPOTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False) or (
            hasattr(self.args, "mm_tunable_parts") and (len(self.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in self.args.mm_tunable_parts or "mm_vision_resampler" in self.args.mm_tunable_parts))
        ):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
        else:
            # super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)
            # print(type(model))
            # from transformers.modeling_utils import unwrap_model
            # print(type(unwrap_model(model)))
            # print(unwrap_model(model).config)
            if self.args.lora_enable:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                from transformers.modeling_utils import unwrap_model

                unwrapped_model = unwrap_model(model)
                self.save_my_lora_ckpt(output_dir, self.args, unwrapped_model)
            else:
                super(LLaVADPOTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVADPOTrainer, self)._save(output_dir, state_dict)
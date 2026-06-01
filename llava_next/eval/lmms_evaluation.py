import argparse
import torch
import os
from tqdm import tqdm
from datetime import timedelta
import copy

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState

import lmms_eval
from lmms_eval.api.model import lmms
from lmms_eval.tasks import TaskManager
from lmms_eval.evaluator import simple_evaluate, evaluate
from lmms_eval.api.registry import register_model
from lmms_eval import utils
from llava_next.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava_next.conversation import conv_templates, SeparatorStyle
from llava_next.model.builder import load_pretrained_model
from llava_next.utils import disable_torch_init
from llava_next.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from lmms_eval.models.llava import Llava

from loguru import logger as eval_logger

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    lmm_obj = llava_paidge(
        model_path=model_path, 
        model_base=args.model_base, 
        model_name=model_name, 
        conv_template=args.conv_mode, 
        batch_size=1
    )

    task_manager = TaskManager()

    results = simple_evaluate( # call simple_evaluate
        model=lmm_obj,
        tasks=args.tasks.split(","),
        num_fewshot=0,
        task_manager=task_manager,
        cli_args=args
    )
    
    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(results=results, samples=samples if args.log_samples else None, datetime_str=datetime_str)

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--tasks", type=str, default="gqa,textvqa")
    parser.add_argument("--log-samples", type=bool, default=True)
    parser.add_argument("--output-path", type=str, default="./logs/")
    args = parser.parse_args()

    eval_model(args)

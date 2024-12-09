# -*- coding: utf-8 -*-
# @Date    : 2024-12-09 11:15:38
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"an example of evaluating llava (not complete)"
# Copy this file to {model_name}_{model_size}.py and implement the `generate` function

import json
import os
from typing import Any

from common.args import vqa_args

# You'd better use transformers to load the model
# however, if you cannot use transformers, you can put the model's codebase in our repo and load it manually
# Refer to the `contributing` section in the readme.md for how to bypass the linter for external codebase
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# load the model
# incomplete: please refer to the codebase of llava to load the model
disable_torch_init()
model = load_pretrained_model("llava-v1.5-7b", None, "name", device="cuda")


# expose `generate` function
def generate(image_paths: list[str], prompts: list[str]) -> list[str]:
    """
    generate the responses of the model on the given image paths and prompts
    Args:
        image_paths: list[str], the paths of the input images in a batch
        prompts: list[str], the user prompts for the model in a batch
    Returns:
        list[str]: the raw responses of the model
        (no need to do post-processing, but prompts should not be included)
    """
    # TODO: implement this
    # You must use batched generation; this will greatly improve the performance, especially for large models
    # Instead of writing `for` loop, try using the batched version of the model's generation function
    ...


# After finishing the implementation, you can run the following command to evaluate the model
# python run.py --module eval.evaluate --eval_model {model_name}_{model_size} --eval_batchsize 4

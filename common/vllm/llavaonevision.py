# -*- coding: utf-8 -*-
# @Date    : 2024-12-13 11:15:38
# @Author  : Zhangtai.Wu (wzt_1824769368@163.com)

import copy
import sys
import warnings

import requests
import torch
from PIL import Image

from common.args import vqa_args
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model

from .base import GenerateModelBase


class GenerateModel(GenerateModelBase):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        model_path = f"models/{model}"
        model_name = "llava_qwen"
        llava_model_args = {"multimodal": True, "attn_implementation": "sdpa"}
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            None,
            model_name,
            False,
            False,
            device_map="auto",
            **llava_model_args,
            ignore_mismatched_sizes=True,
        )
        self.model.eval()
        self.conv_template = "qwen_2"
        self.tokenizer.padding_side = "left"

    def generate(self, image_paths: list, prompts: list) -> list:
        """
        Generate text responses for a batch of image paths and prompts.

        Args:
            image_paths: A list of image file paths.
            prompts: A list of corresponding prompts for each image.

        Returns:
            A list of generated responses (one for each prompt/image pair).
        """
        assert len(image_paths) == len(prompts)
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images_tensor = (
            self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
            .to(self.model.device)
            .to(dtype=torch.float16)
        )
        responses = []
        for i, prompt in enumerate(prompts):
            question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv = copy.deepcopy(conv_templates[self.conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = (
                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)  # type: ignore
                .to(self.model.device)
            )
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids, images=images_tensor[i : i + 1], use_cache=True, **self.kwargs
                )
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            responses.append(output_text[0])

        return responses

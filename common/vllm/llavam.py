# -*- coding: utf-8 -*-
# @Date    : 2024-12-13 11:15:38
# @Author  : Zhangtai.Wu (wzt_1824769368@163.com)

import torch
from PIL import Image

from common.args import vqa_args
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model

from .base import GenerateModelBase


class GenerateModel(GenerateModelBase):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = f"models/{model}"
        model_name = get_model_name_from_path(model_path)
        print(f"Loading model: {model_name}")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, False, False, device=self.device
        )
        self.model.to(self.device)
        self.conv_template = "vicuna_v1"
        self.tokenizer.padding_side = "left"

    def generate(self, image_paths: list, prompts: list) -> list:
        assert len(image_paths) == len(prompts)
        responses = []
        for image_path, prompt in zip(image_paths, prompts):
            images = Image.open(image_path).convert("RGB")
            if type(self.image_processor) is list:
                image_tensor = []
                for p in self.image_processor:
                    p_image = p.preprocess(images, return_tensors="pt")["pixel_values"]
                    if type(p_image) is list:
                        image_tensor.append(
                            p.preprocess(images, return_tensors="pt")["pixel_values"][0]
                            .cuda()
                            .to(dtype=torch.bfloat16)
                        )
                    else:
                        image_tensor.append(
                            p.preprocess(images, return_tensors="pt")["pixel_values"]
                            .cuda()
                            .to(dtype=torch.bfloat16)
                        )
            else:
                image_tensor = (
                    self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"][0]  # type: ignore
                    .cuda()
                    .to(dtype=torch.bfloat16)
                )
            question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = (
                tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)  # type: ignore
                .to(self.device)
            )
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids, images=image_tensor, use_cache=True, **self.kwargs
                )
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            responses.append(output_text[0])

        return responses

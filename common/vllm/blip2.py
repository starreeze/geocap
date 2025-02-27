# -*- coding: utf-8 -*-
# @Date    : 2024-12-13 11:15:38
# @Author  : Zhangtai.Wu (wzt_1824769368@163.com)

from typing import List

import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from .base import GenerateModelBase


class GenerateModel(GenerateModelBase):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        local_model_path = f"models/{model}"
        self.processor = Blip2Processor.from_pretrained(local_model_path, use_fast=False)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            local_model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor.tokenizer.padding_side = "left"

    def generate(self, image_paths: List[str], prompts: List[str]) -> List[str]:
        assert len(image_paths) == len(prompts)
        image = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        inputs = self.processor(image, prompts, return_tensors="pt", padding=True).to(
            device=self.model.device, dtype=torch.float16
        )
        generated_ids = self.model.generate(**inputs, **self.kwargs)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

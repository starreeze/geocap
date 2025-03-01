# -*- coding: utf-8 -*-
# @Date    : 2024-12-13 11:15:38
# @Author  : Zhangtai.Wu (wzt_1824769368@163.com)

from typing import List

import torch
from PIL import Image
from transformers import InstructBlipForConditionalGeneration, InstructBlipProcessor

from .base import GenerateModelBase


class GenerateModel(GenerateModelBase):
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        """
        Initialize the model and processor using pretrained weights from HuggingFace.
        """
        local_model_path = f"/models/{model}"
        self.processor = InstructBlipProcessor.from_pretrained(local_model_path, use_fast=False)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            local_model_path, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor.tokenizer.padding_side = "left"

    def generate(self, image_paths: List[str], prompts: List[str]) -> List[str]:
        assert len(image_paths) == len(prompts)
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        inputs = self.processor(
            images=images, text=prompts, return_tensors="pt", padding=True, truncation=True
        ).to(device=self.model.device, dtype=torch.float16)
        outputs = self.model.generate(**inputs, **self.kwargs)
        generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return generated_texts

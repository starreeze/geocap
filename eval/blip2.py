# -*- coding: utf-8 -*-
# @Date    : 2024-12-13 11:15:38
# @Author  : Zhangtai.Wu (wzt_1824769368@163.com)

from common.args import vqa_args
from eval.base import GenerateModelBase
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from typing import List


class GenerateModel(GenerateModelBase):
    def __init__(self):
        local_model_path = f"model/{vqa_args.eval_model}"
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=10, temperature=0, do_sample=False)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

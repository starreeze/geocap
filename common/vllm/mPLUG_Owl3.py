# from transformers.image_utils import load_image
import gc
import importlib.util
import os
import random
from datetime import datetime
from typing import Any

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    MllamaForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
)

from .base import GenerateModelBase

enable_flash_attn = True


class GenerateModel(GenerateModelBase):

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        device = "cuda"
        self.path = os.path.join("models", model)
        self.device = device
        if not (os.path.exists(self.path) and os.path.isdir(self.path) and len(os.listdir(self.path)) > 0):
            raise ValueError(f"The model spec {model} is not supported!")
        if importlib.util.find_spec("flash_attn") is not None and enable_flash_attn:
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"
        self.model = AutoModel.from_pretrained(
            self.path,
            torch_dtype=torch.half,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)
        # self.processor = self.model.init_processor(self.tokenizer)

    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        output_list = []
        for image_path, prompt in zip(image_paths, prompts):
            processor = self.model.init_processor(self.tokenizer)
            image = Image.open(image_path).convert("RGB")
            messages = [
                {"role": "user", "content": "<|image|> " + prompt},
                {"role": "assistant", "content": ""},
            ]
            inputs = processor(messages, images=[image], video=None)
            inputs.to(self.device)
            generation_kwargs = {"tokenizer": self.tokenizer, "decode_text": True} | self.kwargs
            output_list += self.model.generate(**inputs, **generation_kwargs)
        return output_list

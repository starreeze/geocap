import os
import torch
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import importlib.util
from typing import Any

from common.args import vqa_args
from eval.base import GenerateModelBase
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    Qwen2VLForConditionalGeneration,
    MllamaForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

# from transformers.image_utils import load_image
import gc

from PIL import Image

enable_flash_attn = True


class GenerateModel(GenerateModelBase):
    def __init__(self):
        model_spec = vqa_args.eval_model
        device = "cuda"
        self.path = os.path.join("models", vqa_args.eval_model)
        self.device = device
        if not (os.path.exists(self.path) and os.path.isdir(self.path) and len(os.listdir(self.path)) > 0):
            raise ValueError(f"The model spec {model_spec} is not supported!")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)

    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        batch_inputs = self.tokenizer.apply_chat_template(
            [
                [
                    {
                        "role": "user",
                        "image": Image.open(image_path).convert("RGB"),
                        "content": prompt,
                    }
                ]
                for image_path, prompt in zip(image_paths, prompts)
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            padding_side="left",
        )
        batch_inputs = {_k: _v.to(self.device) for _k, _v in batch_inputs.items()}
        generation_kwargs = {
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": 0.0,
            "top_k": 1,
        }
        with torch.no_grad():
            outputs = self.model.generate(**batch_inputs, **generation_kwargs)[:, batch_inputs["input_ids"].shape[1] :]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

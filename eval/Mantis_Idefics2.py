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
        if not (
            os.path.exists(self.path)
            and os.path.isdir(self.path)
            and len(os.listdir(self.path)) > 0
        ):
            raise ValueError(f"The model spec {model_spec} is not supported!")
        if importlib.util.find_spec("flash_attn") is not None and enable_flash_attn:
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.path, trust_remote_code=True
        ).eval()
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)

    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        # images = [Image.open(image_path) for image_path in image_paths]
        outputs = []
        for image_path, prompt in zip(image_paths, prompts):
            image = Image.open(image_path)
            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            ]
            input_texts = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(images=[image], text=input_texts, return_tensors="pt")
            inputs = {_k: _v.to(self.device) for _k, _v in inputs.items()}
            output = self.model.generate(**inputs, max_new_tokens=32, do_sample=False)[
                :, inputs["input_ids"].shape[1] :
            ]
            outputs += self.processor.batch_decode(output, skip_special_tokens=True)
        return outputs

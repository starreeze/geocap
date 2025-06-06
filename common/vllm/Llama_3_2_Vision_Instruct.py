import importlib.util
import os

import torch
from PIL import Image
from transformers import AutoProcessor, MllamaForConditionalGeneration

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
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation=attn_impl,
        ).eval()
        # self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=True)

    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        images = [Image.open(image_path) for image_path in image_paths]
        messages = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            for prompt in prompts
        ]
        res = []
        for image, message in zip(images, messages):
            input_text = self.processor.apply_chat_template(message, add_generation_prompt=True)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(self.device)
            output = self.model.generate(**inputs, **self.kwargs)[:, inputs["input_ids"].shape[1] :]
            res.append(self.processor.batch_decode(output, skip_special_tokens=True)[0])
        return res

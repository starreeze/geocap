import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import GenerateModelBase


class GenerateModel(GenerateModelBase):

    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        device = "cuda"
        self.path = os.path.join("models", model)
        self.device = device
        if not (os.path.exists(self.path) and os.path.isdir(self.path) and len(os.listdir(self.path)) > 0):
            raise ValueError(f"The model spec {model} is not supported!")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        ).eval()
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path, trust_remote_code=True)

    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        batch_inputs = self.tokenizer.apply_chat_template(
            [
                [{"role": "user", "image": Image.open(image_path).convert("RGB"), "content": prompt}]
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
        with torch.no_grad():
            outputs = self.model.generate(**batch_inputs, **self.kwargs)[
                :, batch_inputs["input_ids"].shape[1] :
            ]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

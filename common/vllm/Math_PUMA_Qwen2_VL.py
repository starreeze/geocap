import importlib.util
import os

import torch
from PIL import Image

from common.args import vqa_args
from qwen2vlm import Qwen2vlmForConditionalGeneration, Qwen2vlmProcessor

from .base import GenerateModelBase

enable_flash_attn = True


class GenerateModel(GenerateModelBase):
    def __init__(self):
        model_spec = vqa_args.eval_model
        device = "cuda"
        self.path = os.path.join("models", vqa_args.eval_model)
        self.device = device
        if not (os.path.exists(self.path) and os.path.isdir(self.path) and len(os.listdir(self.path)) > 0):
            raise ValueError(f"The model spec {model_spec} is not supported!")
        if importlib.util.find_spec("flash_attn") is not None and enable_flash_attn:
            attn_impl = "flash_attention_2"
        else:
            attn_impl = "sdpa"
        self.model = Qwen2vlmForConditionalGeneration.from_pretrained(
            self.path, torch_dtype="auto", attn_implementation=attn_impl, device_map="auto"
        ).eval()
        self.processor = Qwen2vlmProcessor.from_pretrained(self.path)

    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        messages = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            for image_path, prompt in zip(image_paths, prompts)
        ]
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]

        inputs = self.processor(
            texts, [Image.open(path).convert("RGB") for path in image_paths], return_tensors="pt"
        ).to(self.device, torch.bfloat16)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                top_k=1,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return output_text

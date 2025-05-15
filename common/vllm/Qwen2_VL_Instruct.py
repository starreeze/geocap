import importlib.util
import os

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

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
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.path, torch_dtype="auto", attn_implementation=attn_impl, device_map="auto"
        ).eval()
        # self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(self.path)

    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        messages = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt}],
                }
            ]
            for image_path, prompt in zip(image_paths, prompts)
        ]
        texts = [
            self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        # breakpoint()
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.kwargs)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        return output_text

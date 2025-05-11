import importlib.util
import os
from io import BytesIO

import requests
import torch
from common.args import vqa_args
from PIL import Image
from transformers import AutoTokenizer

from llava import LlavaLlamaForCausalLM
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token

from .base import GenerateModelBase


def load_image(image_file):
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_pretrain_model(model_path, device_map="auto", attn_implementation="sdpa"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,  # load_in_8bit=True,
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation,
    )
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    if device_map != "auto":
        vision_tower.to(device=device_map, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


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
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrain_model(
            self.path, device_map=device, attn_implementation=attn_impl
        )
        self.model = self.model.to(self.device).eval()

    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        images = [load_image(image_path) for image_path in image_paths]
        output_texts = []
        for i, image in enumerate(images):
            image_tensor = process_images([image], self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [image.to(self.device, dtype=torch.float16) for image in image_tensor]
            elif isinstance(image_tensor, torch.Tensor):
                image_tensor = image_tensor.to(self.device, dtype=torch.float16)  # pyright: ignore
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###Human: <image>\nHint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\nQuestion:\n{prompts[i].replace('Please directly answer A, B, C or D and nothing else.', '')}###Assistant: "
            input_ids = (
                tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
                .unsqueeze(0)
                .to(self.device)
            )
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=32,
                use_cache=True,
            )
            output_texts.append(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))
        return output_texts

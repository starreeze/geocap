# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/nfs05/xingsy/wzt/geocap/")
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model


class GenerateModel:
    """A class to generate responses using a pre-trained model for images and prompts."""

    def __init__(self, model_path: str):
        model_name = get_model_name_from_path(model_path)
        print(f"Loading model: {model_name}")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, device_map="auto"
        )
        self.conv_template = "vicuna_v1"
        self.tokenizer.padding_side = "left"


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with LLaVA model on MedXpertQA dataset")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="llava-7b")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"gen-{args.model_name}.json")

    print(f"Reading {args.input_file}...")
    data = []
    with open(args.input_file) as f:
        for line in f:
            data.append(json.loads(line))

    model = GenerateModel(model_path=args.model_path)

    responses = []
    correct_count = 0
    for input_sample in tqdm(data, desc="Processing samples"):

        question = input_sample["question"].strip()
        label = input_sample["label"].strip()

        prompt = f"Q: {question}\nA: Please directly answer A, B, C or D and nothing else."

        full_question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates[model.conv_template].copy()
        conv.append_message(conv.roles[0], full_question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt_text, model.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(model.model.device)
        )  # type: ignore

        images = input_sample.get("images", [])
        if images:
            image_paths = [os.path.join(args.image_dir, img) for img in images]
            images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
            images_tensor = (
                model.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
                .to(model.model.device)
                .to(torch.float16)
            )
        else:
            images_tensor = None

        with torch.inference_mode():
            output_ids = model.model.generate(
                input_ids, images=images_tensor, do_sample=False, max_new_tokens=512, use_cache=True
            )

        output_text = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        responses.append(
            {
                "question": question,
                "label": label,
                "response": output_text,
                "images": images if images else None,
            }
        )

        if output_text.strip().lower() == label.lower():
            correct_count += 1

    with open(output_file, "w") as f:
        json.dump(responses, f, indent=4)
    print(f"\nResults saved to {output_file}")

    accuracy = correct_count / len(data)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

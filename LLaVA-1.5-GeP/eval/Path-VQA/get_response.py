# -*- coding: utf-8 -*-
import argparse
import json
import os
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model


class GenerateModel:
    def __init__(self, model_path: str):
        model_name = get_model_name_from_path(model_path)
        print(f"Loading model: {model_name}")
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map="auto"
        )
        self.conv_template = "vicuna_v1"
        self.tokenizer.padding_side = "left"

    def generate(self, images: list, prompts: list) -> list:
        assert len(images) == len(prompts), "Images and prompts must have the same length."

        images_tensor = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        images_tensor = images_tensor.to(self.model.device).to(dtype=torch.float16)

        responses = []
        for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Processing"):
            question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_text = conv.get_prompt()
            input_ids = tokenizer_image_token(
                prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            input_ids = input_ids.unsqueeze(0).to(self.model.device)  # type: ignore

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images_tensor[i : i + 1],
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True,
                )
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            responses.append(output_text[0])

        return responses


def parse_args():
    parser = argparse.ArgumentParser(description="Run VQA evaluation with LLaVA model")
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default="llava-7b")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = GenerateModel(model_path=args.model_path)
    images, prompts, labels = [], [], []
    for file_path in args.input_files:
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing DataFrame"):
            try:
                image_bytes = row["image"]["bytes"]
                if isinstance(image_bytes, pd.Series):
                    image_bytes = image_bytes.values[0]
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                images.append(image)
                prompts.append(f"Q: {row['question']}\nA: ")
                labels.append(row["answer"])
            except Exception as e:
                print(f"Error processing row in {file_path}: {e}")

    responses = model.generate(images, prompts)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"gen-{args.model_name}.json")

    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(
            [{"label": lab, "response": res} for lab, res in zip(labels, responses)],
            outfile,
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    main()

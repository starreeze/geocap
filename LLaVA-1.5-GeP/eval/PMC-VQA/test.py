# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, "/home/nfs05/xingsy/wzt/geocap/")
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model


class GenerateModel:
    """
    A class to generate responses using a pre-trained model for images and prompts.
    """

    def __init__(self, model_path: str):
        model_name = get_model_name_from_path(model_path)
        print(f"Loading model: {model_name}")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, device_map="auto"
        )
        # Set up conversation template
        self.conv_template = "vicuna_v1"
        self.tokenizer.padding_side = "left"

    def generate(self, image_path: str, prompt: str) -> str:
        """
        Generate response for a single image and prompt.

        :param image_path: Path to the input image.
        :param prompt: The textual prompt.
        :return: Generated response.
        """
        # Load image
        image = Image.open(image_path).convert("RGB")
        images_tensor = self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"]
        images_tensor = images_tensor.to(self.model.device).to(dtype=torch.float16)

        # Prepare prompt for conversation
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(self.model.device)  # type: ignore

        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=images_tensor, do_sample=False, max_new_tokens=512, use_cache=True
            )

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output_text[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for image QA tasks")
    parser.add_argument("--csv_path", type=str, default="/PMC-VQA/test_2.csv", help="Path to the CSV file")
    parser.add_argument("--img_dir", type=str, default="/PMC-VQA/figures/", help="Base directory for images")
    parser.add_argument(
        "--model_path", type=str, default="/model/llava-1.5-7b", help="Path to the pretrained model"
    )
    parser.add_argument("--model_name", type=str, default="llava-7b", help="Name of the model")
    parser.add_argument(
        "--output_dir", type=str, default="/PMC-VQA/result", help="Output directory for results"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv_path)
    output_file = os.path.join(args.output_dir, f"gen-{args.model_name}.json")
    model = GenerateModel(model_path=args.model_path)
    correct_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Data"):
        image_path = os.path.join(args.img_dir, str(row["Figure_path"]))
        question = row["Question"]
        choice_a = row["Choice A"]
        choice_b = row["Choice B"]
        choice_c = row["Choice C"]
        choice_d = row["Choice D"]
        prompt = f"Q: {question}\n{choice_a}\n{choice_b}\n{choice_c}\n{choice_d}\nA: Please directly answer A, B, C or D and nothing else."
        response = model.generate(image_path, prompt)
        output_data = {"label": row["Answer"], "response": response}
        with open(output_file, "a", encoding="utf-8") as outfile:
            json.dump(output_data, outfile, ensure_ascii=False, indent=4)
            outfile.write("\n")
        if response.strip().lower() == row["Answer"].lower():
            correct_count += 1
    accuracy = correct_count / len(df)
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()

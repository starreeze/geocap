# -*- coding: utf-8 -*-
import argparse
import json
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

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
        self.conv_template = "vicuna_v1"
        self.tokenizer.padding_side = "left"

    def generate(self, image_paths: list, prompts: list) -> list:
        """
        Generate responses for the given images and prompts.

        :param image_paths: List of paths to input images.
        :param prompts: List of textual prompts.
        :return: List of generated responses.
        """
        assert len(image_paths) == len(prompts), "Image paths and prompts must have the same length."

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        images_tensor = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        images_tensor = images_tensor.to(self.model.device).to(dtype=torch.float16)

        responses = []
        for i, prompt in tqdm(list(enumerate(prompts)), total=len(prompts), desc="Processing Prompts"):
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
                    max_new_tokens=128,
                    use_cache=True,
                )
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            responses.append(output_text[0])

        return responses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize model
    model = GenerateModel(model_path=args.model_path)

    # Read data
    data = pd.read_json(args.input_file)
    image_paths = []
    prompts = []
    labels = []

    # Process data
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing Data"):
        if str(row["q_lang"]) == "en":
            image_path = os.path.join(args.img_dir, str(row["img_name"]))
            image_paths.append(image_path)
            labels.append(row["answer"])

            question = row["question"]
            answer_type = str(row["answer_type"])

        if answer_type == "CLOSED":
            prompt = f"Question: {question}\nPlease directly answer yes or no and nothing else.Answer: "
        else:
            prompt = f"Question: {question}\n Answer:"

        prompts.append(prompt)

    responses = model.generate(image_paths, prompts)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"gen-{args.model_name}.json")
    output_data = [
        {"label": lab, "response": res, "answer_type": atype, "question": q}
        for lab, res, atype, q in zip(labels, responses, data["answer_type"], data["question"])
    ]
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import argparse
import json
import os
from io import BytesIO

import numpy as np
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

    def generate(self, image: Image.Image, prompt: str) -> str:
        """
        Generate response for a single image and prompt.

        :param image: A PIL image object.
        :param prompt: The textual prompt.
        :return: The generated response.
        """
        question = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        conv = conv_templates[self.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(self.model.device)  # type: ignore

        # Process the image
        images_tensor = self.image_processor.preprocess([image], return_tensors="pt")["pixel_values"]
        images_tensor = images_tensor.to(self.model.device).to(dtype=torch.float16)

        # Generate the response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=images_tensor, do_sample=False, max_new_tokens=30, use_cache=True
            )
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return output_text[0]


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_files", nargs="+", required=True)
    parser.add_argument("--output_file", type=str, required=True)

    args = parser.parse_args()

    # Initialize model
    model = GenerateModel(model_path=args.model_path)

    # Generate responses and save results
    responses = []
    labels = []
    output_data = []

    for file_path in tqdm(args.input_files, desc="Processing files"):
        df = pd.read_parquet(file_path)
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_path}", leave=False):
            image_bytes = row["image"]["bytes"]
            image = Image.open(BytesIO(image_bytes)).convert("RGB")  # type: ignore

            question = row["question"]

            label = row["answers"][0]
            prompt = f"Question: {question} \n Answer: "

            response = model.generate(image, prompt)

            responses.append(response)
            labels.append(label)
            output_data.append(
                {
                    "label": label.tolist() if isinstance(label, np.ndarray) else label,
                    "response": response,
                    "question": question,
                }
            )

            del image
            torch.cuda.empty_cache()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Save results
    with open(args.output_file, "w") as outfile:
        json.dump(output_data, outfile, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

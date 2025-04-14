import argparse
import json
import os

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
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, device_map="auto"
        )
        self.conv_template = "vicuna_v1"
        self.tokenizer.padding_side = "left"

    def generate(self, image_paths: list, prompts: list) -> list:
        assert len(image_paths) == len(prompts), "Image paths and prompts must have the same length."

        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
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

    parser = argparse.ArgumentParser(description="Generate model from images and prompts.")
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model.")
    parser.add_argument("--test_file", type=str, help="Path to the test JSON file.")
    parser.add_argument("--image_dir", type=str, help="Path to image directory.")
    parser.add_argument("--response_file", type=str, help="Path to save the output JSON file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = GenerateModel(model_path=args.model_path)
    image_paths = []
    prompts = []

    with open(args.test_file, "r") as f:
        data = json.load(f)

    for item in data:
        prompt = item["query_wo"]
        prompts.append(prompt)
        image_path = os.path.join(args.image_dir, item["image"])
        if os.path.exists(image_path):
            image_paths.append(image_path)
        else:
            print(f"Warning: Image file not found at {image_path}")
    responses = model.generate(image_paths, prompts)
    response_list = []
    for i, response in enumerate(responses):
        data[i]["model_answer"] = response

    with open(args.response_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Responses saved to {args.response_file}")

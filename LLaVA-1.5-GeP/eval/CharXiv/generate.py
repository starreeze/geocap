import os
import json
import argparse
from PIL import Image
import torch
from tqdm import tqdm
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=False,
        default="CharXiv/data",
        help="Directory containing the input json files",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=False,
        default="CharXiv/images",
        help="Directory containing the images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=False,
        default="CharXiv/results",
        help="Directory to save the output json files",
    )
    parser.add_argument(
        "--split", type=str, required=False, choices=["val", "test"], default="val", help="Split of the data"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        choices=["descriptive", "reasoning"],
        default="descriptive",
        help="Mode of the evaluation",
    )

    parser.add_argument("--model_path", type=str, required=False, default="/model/llava-1.5-7b")
    parser.add_argument("--model_name", type=str, required=False, default="llava-1.5-7b")
    args = parser.parse_args()

    input_file = os.path.join(args.data_dir, f"{args.mode}_{args.split}.json")
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)
    model_name = args.model_name

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"gen-{model_name}-{args.mode}_{args.split}.json")
    model = GenerateModel(model_path=args.model_path)

    if args.mode == "descriptive":
        from descriptive_utils import build_descriptive_quries

        queries = build_descriptive_quries(data, args.image_dir)
    elif args.mode == "reasoning":
        from reasoning_utils import build_reasoning_queries

        queries = build_reasoning_queries(data, args.image_dir)
    else:
        raise ValueError("Mode not supported")

    print("Number of test problems to run:", len(queries))
    print("Evaluation mode:", args.mode)
    print("Output file:", output_file)
    prompts = []
    image_paths = []
    for k in queries:
        query = queries[k]["question"]
        image = queries[k]["figure_path"]
        prompts.append(query)
        image_paths.append(image)
    res = model.generate(prompts=prompts, image_paths=image_paths)

    for k in queries:
        queries[k]["response"] = res.pop(0)

    for k in queries:
        queries[k].pop("figure_path", None)
        queries[k].pop("question", None)

    try:
        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(queries, f, indent=4)
        print(f"Results saved.")
    except Exception as e:
        print(e)
        print(f"Error in saving {output_file}")

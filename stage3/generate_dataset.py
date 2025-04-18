import json
import os
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common.args import caption_args, feat_recog_args, logger
from common.llm import generator_mapping, model_path_mapping
from data.caption.paraphrase import Paraphraser
from stage3.recognize import recognize_feature
from stage3.utils import get_circle_points

os.makedirs(feat_recog_args.save_data_path, exist_ok=True)
images_path = "whole_images"
instructions_path = os.path.join(feat_recog_args.save_data_path, "instructions_all.jsonl")
stage3_data_path = os.path.join(feat_recog_args.save_data_path, "num_replace.jsonl")
stage3_paraphrase_path = os.path.join(feat_recog_args.save_data_path, "paraphrase.jsonl")
stage3_tag_format_path = os.path.join(feat_recog_args.save_data_path, "tag_format.jsonl")
stage3_add_default_value_path = os.path.join(feat_recog_args.save_data_path, "add_default_value.jsonl")
llava_data_path = os.path.join(feat_recog_args.save_data_path, "stage3_llava.jsonl")
internvl_data_path = os.path.join(feat_recog_args.save_data_path, "stage3_internvl.jsonl")


class DataGenerator:
    def __init__(self) -> None:
        self.image_path_root = f"{feat_recog_args.fossil_data_path}/{images_path}"
        self.all_images = os.listdir(self.image_path_root)
        self.loaded_llm = False

    def load_llm_generator(self):
        assert not self.loaded_llm
        # Initialize llm
        model_name, model_id = feat_recog_args.num_replace_llm.split("-", 1)
        model_path = model_path_mapping[model_name].format(model_id)
        if "api" in model_name:
            self.llm_generator = generator_mapping[model_name](model_path, max_tokens=4096, temperature=1.0)
        else:
            self.llm_generator = generator_mapping[model_name](model_path, temperature=1.0)
        self.loaded_llm = True
        self.model_name = model_name

        # Initialize prompt
        self.sys_prompt = "You are a helpful assistant."
        with open(feat_recog_args.num_replace_prompt_dir, "r") as f:
            self.user_prompt = f.read()

    def extract_valid_images(self, data_dict: dict):
        # Extract info of Axial image of Holotype specimen
        self.images = []
        for info in data_dict.values():
            images = info["images"]
            for image_dict in images:
                section_type = image_dict["section_type"]
                # specimen_type = image_dict["specimen_type"]
                pixel_mm = image_dict["pixel/mm"]
                if (
                    "axial" in section_type.lower()
                    # and "holotype" in specimen_type.lower()
                    and pixel_mm > 0
                    and image_dict["image"] in self.all_images
                ):
                    image = image_dict["image"]
                    self.images.append({"image": image, "pixel_mm": pixel_mm, "desc": info["desc"]})

    def recognize_features(self, image_info: dict, use_vis_tools: bool = True) -> dict[str, Any]:
        img_path = f"{self.image_path_root}/{image_info['image']}"
        self.img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h, w = self.img.shape[:2]

        new_image_info = image_info
        new_image_info["img_height"] = h
        new_image_info["img_width"] = w

        # Overall shape
        img_countour = self.img[:, :, 3]
        contours, _ = cv2.findContours(img_countour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(contours, key=cv2.contourArea)

        # Get bounding box coordinates
        x_min = min(point[0][0] for point in contour)
        y_min = min(point[0][1] for point in contour)
        x_max = max(point[0][0] for point in contour)
        y_max = max(point[0][1] for point in contour)

        # Calculate length and width/diameter (in mm. ) and ratio
        new_image_info["length"] = (x_max - x_min) * image_info["pixel_mm"]
        new_image_info["width"] = (y_max - y_min) * image_info["pixel_mm"]
        if new_image_info["width"] == 0:
            print(image_info)
        new_image_info["ratio"] = new_image_info["length"] / new_image_info["width"]

        if use_vis_tools:
            # Recognize fossil features
            volutions_dict, thickness_dict, initial_chamber, tunnel_angles = recognize_feature(img_path)

            # Process numerical info
            num_positive_keys = len([k for k in volutions_dict.keys() if k > 0])
            num_negative_keys = len([k for k in volutions_dict.keys() if k < 0])
            larger_key, smaller_key = max(num_positive_keys, num_negative_keys), min(
                num_positive_keys, num_negative_keys
            )
            if larger_key > smaller_key + 1:
                new_image_info["num_volutions"] = larger_key
                # Keep only the keys from the side with more volutions (positive or negative)
                if num_positive_keys > num_negative_keys:
                    volutions_dict = {k: v for k, v in volutions_dict.items() if k > 0}
                else:
                    volutions_dict = {k: v for k, v in volutions_dict.items() if k < 0}
            else:
                new_image_info["num_volutions"] = (num_positive_keys + num_negative_keys) / 2

            # Calculate average height between adjacent volutions
            volution_heights = {}
            for idx, points in volutions_dict.items():
                if idx > 0 and idx - 1 in volutions_dict:
                    next_points = volutions_dict[idx - 1]
                elif idx < 0 and idx + 1 in volutions_dict:
                    next_points = volutions_dict[idx + 1]
                elif idx == 1:
                    initial_chamber_upper = get_circle_points(
                        center=initial_chamber[:2], radius=initial_chamber[2] // 2, angle_range=[225, 315]
                    )
                    next_points = initial_chamber_upper
                elif idx == -1:
                    initial_chamber_lower = get_circle_points(
                        center=initial_chamber[:2], radius=initial_chamber[2] // 2, angle_range=[45, 135]
                    )
                    next_points = initial_chamber_lower
                else:
                    raise ValueError(f"Invalid idx: {idx}")

                y_mean = np.mean([point[1] for point in points])
                next_y_mean = np.mean([point[1] for point in next_points])
                if abs(idx) not in volution_heights:
                    volution_heights[abs(idx)] = abs(y_mean - next_y_mean)
                else:
                    volution_heights[abs(idx)] = (volution_heights[abs(idx)] + abs(y_mean - next_y_mean)) / 2
            # Sort volution_heights by key in ascending order
            volution_heights = dict(sorted(volution_heights.items(), key=lambda item: item[0]))
            new_image_info["volution_heights"] = volution_heights

            # Calculate average thickness and thickness per volutions
            avg_thickness = np.mean([thickness for thickness in thickness_dict.values()])
            new_image_info["avg_thickness"] = avg_thickness * h
            thickness_per_vol = {}
            for idx, thickness in thickness_dict.items():
                if abs(idx) not in thickness_per_vol:
                    thickness_per_vol[abs(idx)] = thickness * h
                else:
                    thickness_per_vol[abs(idx)] = (thickness_per_vol[abs(idx)] + thickness * h) / 2
            thickness_per_vol = dict(sorted(thickness_per_vol.items(), key=lambda item: item[0]))
            new_image_info["thickness_per_vol"] = thickness_per_vol

            if initial_chamber is not None:
                # Convert to diameter
                new_image_info["size_init_chamber"] = initial_chamber[2] * image_info["pixel_mm"] * 1000

            tunnel_angles = self.post_process_tunnel_angles(
                tunnel_angles, int(new_image_info["num_volutions"])
            )
            new_image_info["tunnel_angles"] = tunnel_angles

        return new_image_info

    def post_process_tunnel_angles(
        self, tunnel_angles: dict, num_volutions: int, low_thres: int = 15, high_thres: int = 55
    ) -> dict:
        # Add default value if fail to detect
        for i in range(1, num_volutions + 1):
            if i not in tunnel_angles:
                tunnel_angles[i] = 30

        # Process out of range values
        in_range_angles = [angle for angle in tunnel_angles.values() if low_thres < angle < high_thres]
        if in_range_angles:
            avg_angles = int(sum(in_range_angles) / len(in_range_angles))
        else:
            avg_angles = 30
        for i, angle in tunnel_angles.items():
            if angle < low_thres or angle > high_thres:
                tunnel_angles[i] = avg_angles

        tunnel_angles = dict(sorted(tunnel_angles.items(), key=lambda x: x[0]))
        # Outer volution has bigger tunnel angles
        for i, angle in tunnel_angles.items():
            if i == 1:
                continue
            tunnel_angles[i] = max(angle, tunnel_angles[i - 1] - 5)

        return tunnel_angles

    def _generate_instruction(self, image_info: dict) -> str:
        instruction = "The following is an image of a paleontological fossil, please provide a detailed description for the fossil image. "
        instruction += "Here is some information about the fossil:\n"
        # overall shape
        instruction += f"length: {image_info['length']:.3f} mm. , width(diameter): {image_info['width']:.3f} mm. ratio: {image_info['ratio']:.3f}\n"
        instruction += f"number of volutions(whorls): {image_info['num_volutions']}\n"

        # thickness of spirotheca
        instruction += f"average thickness of spirotheca: {int(image_info['avg_thickness'] * image_info['pixel_mm'] * 1000)} microns\n"
        thickness_micron = [
            str(int(thickness * image_info["pixel_mm"] * 1000))
            for thickness in image_info["thickness_per_vol"].values()
        ]
        instruction += f"thickness by volutions: {', '.join(thickness_micron)} microns\n"

        # heights of volutions
        heights_micron = [
            str(int(height * image_info["pixel_mm"] * 1000))
            for height in image_info["volution_heights"].values()
        ]
        instruction += f"heights of volution/whorl: {', '.join(heights_micron)} microns\n"

        if "size_init_chamber" in image_info:
            instruction += f"initial chamber(proloculus): {int(image_info['size_init_chamber'])} microns\n"

        if "tunnel_angles" in image_info:
            instruction += "tunnel angles: "
            for i, angle in image_info["tunnel_angles"].items():
                if i == 1:
                    instruction += f"{angle} degrees in the 1st volution, "
                elif i == 2:
                    instruction += f"{angle} degrees in the 2nd volution, "
                elif i == 3:
                    instruction += f"{angle} degrees in the 3rd volution, "
                else:
                    instruction += f"{angle} degrees in the {i}th volution, "
            instruction = instruction.rstrip(", ") + ".\n"

        return instruction

    def _generate_naive_instruction(self, image_info: dict):
        instruction = "The following is an image of a paleontological fossil, please provide a detailed description for the fossil image. "

        width_mm = image_info["img_width"] * image_info["pixel_mm"]
        height_mm = image_info["img_height"] * image_info["pixel_mm"]
        instruction += f"The resolution of the fossil image is {image_info['img_width']}\u00D7{image_info['img_height']}, "  # \u00D7 -> ×
        instruction += f"and the width and height of the image are {width_mm:.3f} mm and {height_mm:.3f} mm, respectively."

        return instruction

    def generate_instructions(self, data_dict: dict, use_vis_tools: bool = True) -> list[dict[str, str]]:
        self.extract_valid_images(data_dict)

        instructions = []
        logger.info("Generating instructions & recognizing features")
        for image_info in tqdm(self.images):
            new_image_info = self.recognize_features(image_info, use_vis_tools)
            if use_vis_tools:
                instruction = self._generate_instruction(new_image_info)
            else:
                instruction = self._generate_naive_instruction(new_image_info)
            sample = {"image": image_info["image"], "input": instruction, "output": image_info["desc"]}
            instructions.append(sample)

        return instructions

    def generate_outputs(self, instructions) -> list[dict[str, str]]:
        if not self.loaded_llm:
            self.load_llm_generator()
        bs = feat_recog_args.num_replace_batchsize

        dataset = []
        messages = []
        # Preprocessing input messages
        for sample in instructions:
            dataset.append({"image": sample["image"], "input": sample["input"], "output": ""})

            instruction = sample["input"]
            desc = sample["output"]
            numerical_info = instruction.split("\n", 1)[1]
            user_prompt = self.user_prompt.replace("{desc}", desc)
            user_prompt = user_prompt.replace("{numerical_info}", numerical_info)
            if "api" in self.model_name:
                messages.append([{"role": "user", "content": user_prompt}])
            else:
                messages.append(
                    [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": user_prompt}]
                )

        num_batches = (len(instructions) + bs - 1) // bs
        responses = self.llm_generator(messages, bs)
        logger.info("Generating outputs using LLM")
        for b, batch in tqdm(enumerate(responses), total=num_batches):
            for i, response in enumerate(batch):
                processed_desc = response
                dataset[b * bs + i]["output"] = processed_desc

        return dataset


def generate_dataset(start_pos=None, end_pos=None, use_vis_tools: bool = True):
    data_path = os.path.join(feat_recog_args.fossil_data_path, "filtered_data.json")
    with open(data_path, "r") as f:
        data_dict = json.load(f)

    dataset = []
    data_generator = DataGenerator()

    # Recognize features and generate instructions
    if not os.path.exists(instructions_path):
        instructions = data_generator.generate_instructions(data_dict, use_vis_tools)
        with open(instructions_path, "w") as f:
            for instruction in instructions:
                f.write(json.dumps(instruction) + "\n")

    # Replace numerical infos in outputs
    with open(instructions_path, "r") as f:
        instructions = [json.loads(line) for line in f]

    if start_pos is None and end_pos is None:
        output_path = stage3_data_path
        start_pos = 0
        end_pos = len(instructions)
    else:
        output_path = os.path.join(feat_recog_args.save_data_path, f"stage3_{start_pos}_{end_pos}.jsonl")

    dataset = data_generator.generate_outputs(instructions[start_pos:end_pos])

    # Save dataset as jsonl file
    with open(output_path, "w") as f:
        for data in dataset:
            f.write(json.dumps(data) + "\n")
    # Release GPU memory occupied by llm_generator
    torch.cuda.empty_cache()


def paraphrase(start_pos=None, end_pos=None):
    paraphraser = Paraphraser()
    # Read original captions
    if start_pos is not None and end_pos is not None:
        input_data_path = os.path.join(feat_recog_args.save_data_path, f"stage3_{start_pos}_{end_pos}.jsonl")
        output_path = os.path.join(
            feat_recog_args.save_data_path, f"stage3_paraphrase_{start_pos}_{end_pos}.jsonl"
        )
    else:
        input_data_path = stage3_data_path
        output_path = stage3_paraphrase_path

    with open(input_data_path, "r") as f:
        captions = [json.loads(line) for line in f]

    # Extract and paraphrase outputs
    original_outputs = [caption["output"] for caption in captions]
    paraphrased_outputs = paraphraser(original_outputs)

    # Replace original outputs with paraphrased ones
    for caption, paraphrased_output in zip(captions, paraphrased_outputs):
        caption["output"] = paraphrased_output

    with open(output_path, "w") as f:
        for caption in captions:
            f.write(json.dumps(caption) + "\n")
    torch.cuda.empty_cache()


def tag_format(start_pos=None, end_pos=None):
    caption_args.paraphrase_prompt_dir = "data/caption/prompts/tag_format.txt"
    paraphraser = Paraphraser()
    # Read original captions
    if start_pos is not None and end_pos is not None:
        input_data_path = os.path.join(
            feat_recog_args.save_data_path, f"stage3_paraphrase_{start_pos}_{end_pos}.jsonl"
        )
        output_path = os.path.join(
            feat_recog_args.save_data_path, f"stage3_tag_format_{start_pos}_{end_pos}.jsonl"
        )
    else:
        input_data_path = stage3_paraphrase_path
        output_path = stage3_tag_format_path

    with open(input_data_path, "r") as f:
        captions = [json.loads(line) for line in f]

    # Extract and paraphrase outputs
    original_outputs = [caption["output"] for caption in captions]
    paraphrased_outputs = paraphraser(original_outputs)

    # Replace original outputs with paraphrased ones
    for caption, paraphrased_output in zip(captions, paraphrased_outputs):
        caption["output"] = paraphrased_output

    with open(output_path, "w") as f:
        for caption in captions:
            f.write(json.dumps(caption) + "\n")
    torch.cuda.empty_cache()


def add_default_value(start_pos=None, end_pos=None):
    caption_args.paraphrase_prompt_dir = "data/caption/prompts/add_default_value.txt"
    paraphraser = Paraphraser()
    # Read original captions
    if start_pos is not None and end_pos is not None:
        input_data_path = os.path.join(
            feat_recog_args.save_data_path, f"stage3_paraphrase_{start_pos}_{end_pos}.jsonl"
        )
        output_path = os.path.join(
            feat_recog_args.save_data_path, f"stage3_tag_format_{start_pos}_{end_pos}.jsonl"
        )
    else:
        input_data_path = stage3_tag_format_path
        output_path = stage3_add_default_value_path

    with open(input_data_path, "r") as f:
        captions = [json.loads(line) for line in f]

    # Extract and paraphrase outputs
    original_outputs = [caption["output"] for caption in captions]
    paraphrased_outputs = paraphraser(original_outputs)

    # Replace original outputs with paraphrased ones
    for caption, paraphrased_output in zip(captions, paraphrased_outputs):
        caption["output"] = paraphrased_output

    with open(output_path, "w") as f:
        for caption in captions:
            f.write(json.dumps(caption) + "\n")
    torch.cuda.empty_cache()


def format_to_llava():
    data: list[str] = open(stage3_paraphrase_path, "r").readlines()
    target_data = []
    for i, d in enumerate(data):
        d = json.loads(d.strip())
        target_data.append(
            {
                "id": i,
                "image": d["image"],
                "conversations": [
                    {"from": "human", "value": "<image>\n" + d["input"]},
                    {"from": "gpt", "value": d["output"]},
                ],
            }
        )
    json.dump(target_data, open(llava_data_path, "w"), indent=2)


def format_to_internvl():
    # data: list[str] = open(stage3_paraphrase_path, "r").readlines()
    data: list[str] = open(stage3_tag_format_path, "r").readlines()
    with open(internvl_data_path, "w") as f:
        for i, d in enumerate(data):
            d = json.loads(d.strip())
            target_data = {
                "id": i,
                "image": d["image"],
                "conversations": [
                    {"from": "human", "value": "<image>\n" + d["input"]},
                    {"from": "gpt", "value": d["output"]},
                ],
            }
            f.write(json.dumps(target_data) + "\n")


def main():
    # generate_dataset(use_vis_tools=True)
    paraphrase()
    tag_format()
    add_default_value()
    # format_to_llava()
    format_to_internvl()


if __name__ == "__main__":
    main()

import json
import os
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from common.args import caption_args, feat_recog_args, logger
from common.llm import generator_mapping, model_path_mapping
from data.caption.paraphrase import Paraphraser
from stage3.get_angles_and_slope import get_angles_and_slope
from stage3.recognize import recognize_feature
from stage3.reorder_tag import reorder_tag
from stage3.utils import get_circle_points

os.makedirs(feat_recog_args.save_data_path, exist_ok=True)
images_path = "whole_images"
instructions_path = os.path.join(feat_recog_args.save_data_path, "instructions_all.jsonl")
stage3_data_path = os.path.join(feat_recog_args.save_data_path, "num_replace.jsonl")
stage3_paraphrase_path = os.path.join(feat_recog_args.save_data_path, "paraphrase.jsonl")
stage3_tag_format_path = os.path.join(feat_recog_args.save_data_path, "tag_format.jsonl")
stage3_add_default_value_path = os.path.join(feat_recog_args.save_data_path, "add_default_value.jsonl")
stage3_reorder_tag_path = os.path.join(feat_recog_args.save_data_path, "reorder_tag.jsonl")
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

        # Classify size by length
        if new_image_info["length"] < 1:
            new_image_info["size"] = "minute"
        elif new_image_info["length"] < 3:
            new_image_info["size"] = "small"
        elif new_image_info["length"] < 6:
            new_image_info["size"] = "medium"
        elif new_image_info["length"] < 10:
            new_image_info["size"] = "large"
        elif new_image_info["length"] < 20:
            new_image_info["size"] = "mega"
        else:
            new_image_info["size"] = "gigantic"

        # Get equator, slope and poles
        angles_and_scores = get_angles_and_slope(img_path)
        left_angle, right_angle, upper_angle, lower_angle = angles_and_scores[:4]
        equator_angle = (upper_angle + lower_angle) / 2
        if equator_angle < 2.95:
            equator = "convex"
        elif equator_angle < 3.15:
            equator = "straight"
        else:
            equator = "concave"
        new_image_info["equator"] = equator

        poles_angle = (left_angle + right_angle) / 2
        if poles_angle < 2.35:
            poles = "pointed"
        else:
            poles = "blunted"
        new_image_info["poles"] = poles

        slopes_score = np.sum(angles_and_scores[4:])
        if slopes_score < -2.3:
            slopes = "convex"
        elif slopes_score < -0.8:
            slopes = "straight"
        else:
            slopes = "concave"
        new_image_info["slopes"] = slopes

        # Classify shape by ratio and contour
        # shape_type = classify_ellipsoidal_vs_fusiform(contour)  # "ellipsoidal" or "fusiform"
        shape_type = "ellipsoidal" if slopes == "convex" else "fusiform"
        ellipsoidal_classes = {
            "prolate spherical": [0.9, 0.98],
            "spherical": [0.98, 1.05],
            "sub-spherical": [1.05, 1.11],
            "ellipsoidal": [1.11, 3],
            "elongate ellipsoidal": [3, 6],
            "cylindrical": [6, 999],
        }
        fusiform_classes = {
            "lentoid": [0, 0.75],
            "rhombus": [0.75, 1.3],
            "inflated fusiform": [1.3, 1.9],
            "fusiform": [1.9, 3.5],
            "elongate fusiform": [3.5, 999],
        }
        ratio2shape = ellipsoidal_classes if shape_type == "ellipsoidal" else fusiform_classes
        for shape, ratio_range in ratio2shape.items():
            if ratio_range[0] < new_image_info["ratio"] <= ratio_range[1]:
                new_image_info["shape"] = shape
                break

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

            # Average tunnel angles by volutions
            tunnel_angle = int(sum(tunnel_angles.values()) / len(tunnel_angles))
            new_image_info["tunnel_angle"] = tunnel_angle

        return new_image_info

    def post_process_tunnel_angles(
        self, tunnel_angles: dict, num_volutions: int, low_thres: int = 15, high_thres: int = 55
    ) -> dict:
        # Add default value if fail to detect
        for i in range(1, num_volutions + 1):
            if i not in tunnel_angles:
                tunnel_angles[i] = 25

        # Process out of range values
        in_range_angles = [angle for angle in tunnel_angles.values() if low_thres < angle < high_thres]
        if in_range_angles:
            avg_angles = int(sum(in_range_angles) / len(in_range_angles))
        else:
            avg_angles = 25
        for i, angle in tunnel_angles.items():
            if angle < low_thres or angle > high_thres:
                tunnel_angles[i] = avg_angles

        tunnel_angles = dict(sorted(tunnel_angles.items(), key=lambda x: x[0]))

        # Create a more reliable baseline based on all angles
        # Calculate a smooth increasing trend for tunnel angles
        min_angle = min(tunnel_angles.values())
        max_angle = max(tunnel_angles.values())
        total_volutions = max(tunnel_angles.keys())

        # If there's a clear increasing trend in the data, preserve it
        # Otherwise, apply a gentle increasing gradient
        if max_angle > min_angle and total_volutions > 1:
            # Calculate average increase per volution
            avg_increase = (max_angle - min_angle) / (total_volutions - 1)

            # Apply smoothed values
            base_angle = min(tunnel_angles.values())
            for i in tunnel_angles.keys():
                # Gradually increase angle with volution number
                expected_angle = base_angle + (i - 1) * avg_increase
                # Blend original and expected values (50% original, 50% expected)
                tunnel_angles[i] = int(0.5 * tunnel_angles[i] + 0.5 * expected_angle)

        return tunnel_angles

    def get_one_instruction(self, image_info: dict) -> str:
        instruction = "The following is an image of a paleontological fossil, please provide a detailed description for the fossil image. "
        instruction += "Here is some information about the fossil that must be included in the description:\n"
        # size and shape
        instruction += f"size: {image_info['size']}, shape: {image_info['shape']}, "

        # equator, slopes and poles
        instruction += f"equator(central part): {image_info['equator']}, "
        instruction += f"slopes: {image_info['slopes']}, "
        instruction += f"poles: {image_info['poles']}\n"

        # length, width, ratio
        instruction += f"length: {image_info['length']:.3f} mm. , width(diameter): {image_info['width']:.3f} mm. ratio: {image_info['ratio']:.3f}\n"
        # number of volutions
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

        if "tunnel_angle" in image_info:
            instruction += f"tunnel angle: {image_info['tunnel_angle']} degrees.\n"

        return instruction

    def get_one_novis_instruction(self, image_info: dict):
        instruction = "The following is an image of a paleontological fossil, please provide a detailed description for the fossil image. "

        width_mm = image_info["img_width"] * image_info["pixel_mm"]
        height_mm = image_info["img_height"] * image_info["pixel_mm"]
        instruction += f"The resolution of the fossil image is {image_info['img_width']}\u00d7{image_info['img_height']}, "  # \u00D7 -> ×
        instruction += f"and the width and height of the image are {width_mm:.3f} mm and {height_mm:.3f} mm, respectively."

        return instruction

    def generate_instructions(self, data_dict: dict, use_vis_tools: bool = True) -> list[dict[str, str]]:
        self.extract_valid_images(data_dict)

        instructions = []
        logger.info("Generating instructions & recognizing features")
        for image_info in tqdm(self.images):
            new_image_info = self.recognize_features(image_info, use_vis_tools)
            if use_vis_tools:
                instruction = self.get_one_instruction(new_image_info)
            else:
                instruction = self.get_one_novis_instruction(new_image_info)
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
    """
    Generate initial dataset with feature recognition and instruction generation
    """
    data_path = os.path.join(feat_recog_args.fossil_data_path, "filtered_data.json")
    with open(data_path, "r") as f:
        data_dict = json.load(f)

    data_generator = DataGenerator()

    # Step 1: Recognize features and generate instructions
    if not os.path.exists(instructions_path):
        instructions = data_generator.generate_instructions(data_dict, use_vis_tools)
        with open(instructions_path, "w") as f:
            for instruction in instructions:
                f.write(json.dumps(instruction) + "\n")

    # Read instructions
    with open(instructions_path, "r") as f:
        instructions = [json.loads(line) for line in f]

    # Generate outputs for the instructions
    if start_pos is None and end_pos is None:
        start_pos = 0
        end_pos = len(instructions)
        output_path = stage3_data_path
    else:
        output_path = os.path.join(feat_recog_args.save_data_path, f"stage3_{start_pos}_{end_pos}.jsonl")

    selected_instructions = instructions[start_pos:end_pos]
    dataset = data_generator.generate_outputs(selected_instructions)

    # Save dataset
    with open(output_path, "w") as f:
        for data in dataset:
            f.write(json.dumps(data) + "\n")

    return dataset


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


def tag_format(start_pos=None, end_pos=None):
    caption_args.paraphrase_prompt_dir = "stage3/prompts/tag_format.txt"
    paraphraser = Paraphraser()
    # Read original captions
    if start_pos is not None and end_pos is not None:
        input_data_path = os.path.join(
            feat_recog_args.save_data_path, f"paraphrase_{start_pos}_{end_pos}.jsonl"
        )
        output_path = os.path.join(feat_recog_args.save_data_path, f"tag_format_{start_pos}_{end_pos}.jsonl")
    else:
        input_data_path = stage3_paraphrase_path
        output_path = stage3_tag_format_path

    with open(input_data_path, "r") as f:
        captions = [json.loads(line) for line in f]

    # Process in batches and write incrementally
    batch_size = caption_args.caption_batchsize
    with open(output_path, "a") as f:
        for i in tqdm(range(0, len(captions), batch_size), desc="Tag Format"):
            batch_captions = captions[i : i + batch_size]
            batch_outputs = [caption["output"] for caption in batch_captions]

            # Process this batch
            paraphrased_batch = paraphraser(batch_outputs)

            # Write this batch to file immediately
            for caption, paraphrased_output in zip(batch_captions, paraphrased_batch):
                caption["output"] = paraphrased_output
                f.write(json.dumps(caption) + "\n")
            f.flush()


def add_default_value(start_pos=None, end_pos=None):
    caption_args.paraphrase_prompt_dir = "stage3/prompts/add_default_value.txt"
    paraphraser = Paraphraser()
    # Read original captions
    if start_pos is not None and end_pos is not None:
        input_data_path = os.path.join(
            feat_recog_args.save_data_path, f"tag_format_{start_pos}_{end_pos}.jsonl"
        )
        output_path = os.path.join(
            feat_recog_args.save_data_path, f"add_default_value_{start_pos}_{end_pos}.jsonl"
        )
    else:
        input_data_path = stage3_tag_format_path
        output_path = stage3_add_default_value_path

    with open(input_data_path, "r") as f:
        captions = [json.loads(line) for line in f]

    # Process in batches and write incrementally
    batch_size = caption_args.caption_batchsize
    with open(output_path, "a") as f:
        for i in tqdm(range(0, len(captions), batch_size), desc="Add Default Value"):
            batch_captions = captions[i : i + batch_size]
            batch_outputs = [caption["output"] for caption in batch_captions]

            # Process this batch
            paraphrased_batch = paraphraser(batch_outputs)

            # Write this batch to file immediately
            for caption, paraphrased_output in zip(batch_captions, paraphrased_batch):
                caption["output"] = paraphrased_output
                f.write(json.dumps(caption) + "\n")
            f.flush()


def numerical_replacement(start_pos=None, end_pos=None):
    """
    Perform numerical replacement using LLM
    """
    data_generator = DataGenerator()

    # Read data with default values added
    if start_pos is not None and end_pos is not None:
        input_data_path = os.path.join(
            feat_recog_args.save_data_path, f"add_default_value_{start_pos}_{end_pos}.jsonl"
        )
        output_path = os.path.join(feat_recog_args.save_data_path, f"stage3_{start_pos}_{end_pos}.jsonl")
    else:
        input_data_path = stage3_add_default_value_path
        output_path = stage3_data_path

    with open(input_data_path, "r") as f:
        instructions = [json.loads(line) for line in f]

    # Generate outputs using LLM for numerical replacement
    dataset = data_generator.generate_outputs(instructions)

    # Save final dataset
    with open(output_path, "w") as f:
        for data in dataset:
            f.write(json.dumps(data) + "\n")


def format_to_llava():
    data: list[str] = open(stage3_data_path, "r").readlines()
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
    data: list[str] = open(stage3_reorder_tag_path, "r", encoding="utf-8").readlines()
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
    """
    Main data processing pipeline with separated steps:
    1. Feature recognition and instruction generation
    2. Paraphrase (text rewriting)
    3. Tag format (adding tags)
    4. Add default values (adding default values)
    5. Numerical replacement (numerical substitution)
    6. Reorder tags (tag reordering)
    7. Format conversion (format conversion)
    """
    # generate_dataset(use_vis_tools=True)
    # paraphrase()
    # tag_format()
    # add_default_value()
    numerical_replacement()
    reorder_tag(input_path=stage3_data_path, output_path=stage3_reorder_tag_path)

    # Step 7: Format conversion
    # format_to_llava()
    format_to_internvl()


if __name__ == "__main__":
    main()

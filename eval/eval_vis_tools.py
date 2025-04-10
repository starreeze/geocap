import json
import os
from typing import Any

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common.args import feat_recog_args, fossil_eval_args, logger
from common.llm import generator_mapping, model_path_mapping
from feat_recognize.recognize import recognize_feature


class VisToolOutputGenerator:
    def __init__(self) -> None:
        self.image_path_root = os.path.join(feat_recog_args.fossil_data_path, "filtered_images")
        self.all_images = os.listdir(self.image_path_root)
        self.loaded_llm = False

    def extract_valid_images(self, data_dict: dict):
        # Extract info of Axial image of Holotype specimen
        self.images = []
        for info in data_dict.values():
            images = info["images"]
            for image_dict in images:
                section_type = image_dict["section_type"]
                specimen_type = image_dict["specimen_type"]
                pixel_mm = image_dict["pixel/mm"]
                if (
                    "axial" in section_type.lower()
                    # and "holotype" in specimen_type.lower()
                    and pixel_mm > 0
                    and image_dict["image"] in self.all_images
                ):
                    image = image_dict["image"]
                    self.images.append({"image": image, "pixel_mm": pixel_mm, "desc": info["desc"]})

    def recognize_features(self, image_info: dict) -> dict[str, Any]:
        img_path = os.path.join(self.image_path_root, image_info["image"])
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

        # Recognize fossil features
        volutions_dict, thickness_dict, initial_chamber, tunnel_angles = recognize_feature(img_path)

        # Process numerical info
        has_positive = any(k > 0 for k in volutions_dict.keys())
        has_negative = any(k < 0 for k in volutions_dict.keys())
        if has_positive and has_negative:
            new_image_info["num_volutions"] = len(volutions_dict) / 2
        else:
            new_image_info["num_volutions"] = len(volutions_dict)

        # Calculate average height between adjacent volutions
        volution_heights = {}
        for idx, points in volutions_dict.items():
            if idx > 0 and idx - 1 in volutions_dict:
                next_points = volutions_dict[idx - 1]
            elif idx < 0 and idx + 1 in volutions_dict:
                next_points = volutions_dict[idx + 1]
            else:
                continue
            y_mean = np.mean([point[1] for point in points])
            next_y_mean = np.mean([point[1] for point in next_points])
            if abs(idx) - 1 not in volution_heights:
                volution_heights[abs(idx) - 1] = abs(y_mean - next_y_mean)
            else:
                volution_heights[abs(idx) - 1] = (
                    volution_heights[abs(idx) - 1] + abs(y_mean - next_y_mean)
                ) / 2
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

        if tunnel_angles:
            new_image_info["tunnel_angles"] = tunnel_angles

        return new_image_info


def generate_vis_tools_output():
    # Create a data generator instance
    evaluator = VisToolOutputGenerator()

    # Load data from a JSON file containing fossil information
    data_path = os.path.join(feat_recog_args.fossil_data_path, "filtered_data.json")
    with open(data_path, "r") as f:
        data_dict = json.load(f)

    evaluator.extract_valid_images(data_dict)

    img_2_pixel_mm = {image_info["image"]: image_info["pixel_mm"] for image_info in evaluator.images}

    # Load test image files
    test_image_path = "dataset/instructions_no_vis_tools/instructions_test.jsonl"
    with open(test_image_path, "r") as f:
        test_images = [json.loads(line)["image"] for line in f]

    # Process each image and extract required information
    output_info = []
    logger.info("Processing images and extracting information")
    for image in tqdm(test_images):
        # Recognize features from the image
        image_info = {"image": image, "pixel_mm": img_2_pixel_mm[image]}
        processed_info = evaluator.recognize_features(image_info)

        # Create a dictionary with the required fields
        fossil_info = {
            "image_path": image,
            "overall_size": "",
            "overall_shape": "",
            "length": f"{processed_info['length']:.3f} mm",
            "width": f"{processed_info['width']:.3f} mm",
            "ratio": f"{processed_info['ratio']:.3f}",
            "axis_shape": "",
            "number_of_volutions": f"{processed_info['num_volutions']}",
            "thickness_of_spirotheca": f"",
            "height_of_volution": "",
            "proloculus": "",
            "tunnel_angles": "",
            "tunnel_shape": "",
            "chomata": "",
            "axial_filling": "",
        }

        # Process thickness of spirotheca
        if "thickness_per_vol" in processed_info:
            thickness_microns = [
                str(int(thickness * processed_info["pixel_mm"] * 1000))
                for thickness in processed_info["thickness_per_vol"].values()
            ]
            fossil_info["thickness_of_spirotheca"] = ", ".join(thickness_microns) + " microns"

        # Process heights of volutions
        if "volution_heights" in processed_info:
            heights_microns = [
                str(int(height * processed_info["pixel_mm"] * 1000))
                for height in processed_info["volution_heights"].values()
            ]
            fossil_info["height_of_volution"] = ", ".join(heights_microns) + " microns"

        # Process proloculus (initial chamber)
        if "size_init_chamber" in processed_info:
            fossil_info["proloculus"] = f"{int(processed_info['size_init_chamber'])} microns"

        # Process tunnel angles
        if "tunnel_angles" in processed_info:
            angles = []
            for i, angle in processed_info["tunnel_angles"].items():
                angles.append(str(angle))
            fossil_info["tunnel_angles"] = ", ".join(angles) + " degrees"

        output_info.append(fossil_info)

    # Save the extracted information to a JSON file
    output_dir = os.path.join(fossil_eval_args.eval_result_dir, "extracted_output_info.json")
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w") as f:
        json.dump(output_info, f, indent=4)

    logger.info(f"Extracted information saved to {output_dir}")


def main():
    generate_vis_tools_output()
    # eval_vis_tools()


if __name__ == "__main__":
    main()

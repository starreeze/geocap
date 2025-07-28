import json
import os

from PIL import Image

if __name__ == "__main__":
    with open("dataset/common/data.json", "r") as f:
        data = json.load(f)

    img2pixel_mm = {}
    for fos_name, fos_dict in data.items():
        for image_dict in fos_dict["images"]:
            img2pixel_mm[image_dict["image"]] = image_dict["pixel/mm"]

    whole_images_dir = "dataset/common/complete_images"
    all_images = os.listdir(whole_images_dir)

    size_info = []
    for image_name in all_images:
        image = Image.open(os.path.join(whole_images_dir, image_name))
        width, height = image.size
        if image_name in img2pixel_mm:
            pixel_mm = img2pixel_mm[image_name]
        else:
            pixel_mm = 0.0
        if pixel_mm == 0.0:
            continue
        width_mm = width * pixel_mm
        height_mm = height * pixel_mm
        info = {"image_path": image_name, "length": width_mm, "width": height_mm}
        size_info.append(info)

    with open("dataset/common/extracted_info/size_info.json", "w") as f:
        json.dump(size_info, f, indent=2)

import json
import os

from tqdm import tqdm

from stage3.get_angles_and_slope import get_angles_and_slope


def main():
    images_dir = "dataset/common/testset_images"
    all_images = os.listdir(images_dir)
    all_images = [image_name for image_name in all_images if image_name.endswith(".png")]

    angles_info_list = []
    slopes_info_list = []
    for image_name in tqdm(all_images):
        path = os.path.join(images_dir, image_name)
        results = get_angles_and_slope(path=path, debug=False)
        left_angle, right_angle, upper_angle, lower_angle = results[:4]
        angles_info = {
            "image_path": image_name,
            "left_angle": left_angle,
            "right_angle": right_angle,
            "upper_angle": upper_angle,
            "lower_angle": lower_angle,
        }
        angles_info_list.append(angles_info)

        convex_scores = results[4:]
        slopes_info = {"image_path": image_name, "convex_scores": list(convex_scores)}
        slopes_info_list.append(slopes_info)

    with open("dataset/common/extracted_info/angles_info.json", "w") as f:
        json.dump(angles_info_list, f, indent=2)

    with open("dataset/common/extracted_info/slopes_info.json", "w") as f:
        json.dump(slopes_info_list, f, indent=2)


if __name__ == "__main__":
    main()

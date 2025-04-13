import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from common.args import feat_recog_args
from feat_recognize.initial_chamber import ProloculusDetector
from feat_recognize.recognize import chomatas_scan
from feat_recognize.utils import resize_img
from feat_recognize.volution_counter import VolutionCounter


def visualize_volutions(
    img_paths,
    output_dir,
    feat_recog_args,
    show_initial_chamber=True,
    show_volution_lines=True,
    show_volution_numbers=True,
    save_format="png",
    dpi=300,
):
    """
    Visualize volutions and initial chamber for a batch of images.

    Parameters:
    img_paths (list): List of paths to input images
    output_dir (str): Directory to save visualization results
    feat_recog_args: Arguments for VolutionCounter
    show_initial_chamber (bool): Whether to visualize the initial chamber
    show_volution_lines (bool): Whether to visualize volution lines
    show_volution_numbers (bool): Whether to show volution numbers
    save_format (str): Format to save images ('png', 'jpg', etc.)
    dpi (int): DPI for saved images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize VolutionCounter
    counter = VolutionCounter(feat_recog_args)
    # Visualize initial chamber
    proloculus_detector = ProloculusDetector()

    for img_path in img_paths:
        # Get filename without extension
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # Read and preprocess image
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        orig_h, orig_w = img.shape[:2]
        orig_img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        img_rgb = resize_img(orig_img_rgb)
        h, w = img_rgb.shape[:2]
        orig_img_gray = cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

        # Detect initial chamber
        initial_chamber = proloculus_detector.detect_initial_chamber(img_path)

        # Count volutions
        if initial_chamber is None:
            center = (w // 2, h // 2)
        else:
            center = tuple(initial_chamber[:2])

        volutions_dict, thickness_dict = counter.count_volutions(img_path, center=center)

        for idx, volution in volutions_dict.items():
            for i, point in enumerate(volution):
                volution[i] = (int(point[0] * orig_w), int(point[1] * orig_h))
            # Remove duplicate x values by keeping only one point per x coordinate
            unique_x_points = {}
            for point in volution:
                x = point[0]
                if x not in unique_x_points:
                    unique_x_points[x] = point
            volution = list(unique_x_points.values())
            volutions_dict[idx] = volution

        chomata_result = chomatas_scan(volutions_dict, img_path, initial_chamber=center)
        for idx, chomata_pos in chomata_result.items():
            if len(chomata_pos) > 1:  # successfully detect 2 chomatas in a voludion
                cv2.circle(orig_img_rgb, chomata_pos[0][:2], 2, (0, 0, 255), 2)
                cv2.circle(orig_img_rgb, chomata_pos[1][:2], 2, (0, 0, 255), 2)

        # Save original image without any markings
        original_output_path = os.path.join(output_dir, f"{img_name}_original.{save_format}")
        plt.figure(figsize=(12, 8))
        plt.imshow(orig_img_rgb)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(original_output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        # Create figure for visualization on binarized image
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get binarized image from counter
        binary_img = cv2.adaptiveThreshold(
            orig_img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2
        )
        binary_img_rgb = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)

        # Display binarized image
        ax.imshow(binary_img_rgb, cmap="gray")

        # Get image dimensions for denormalization
        h, w = img_rgb.shape[:2]

        # Visualize volution lines
        if show_volution_lines:
            colors = plt.cm.jet(np.linspace(0, 1, len(volutions_dict)))

            for i, (vol_idx, points) in enumerate(volutions_dict.items()):
                # Denormalize coordinates
                x_points = [p[0] for p in points]
                y_points = [p[1] for p in points]

                # Plot volution line
                ax.plot(x_points, y_points, "-", color=colors[i], linewidth=2, label=f"Volution {vol_idx}")

                # Add volution number
                # if show_volution_numbers and len(x_points) > 0:
                #     mid_idx = len(x_points) // 2
                #     ax.text(
                #         x_points[mid_idx],
                #         y_points[mid_idx],
                #         str(vol_idx),
                #         color="white",
                #         fontsize=12,
                #         fontweight="bold",
                #         bbox=dict(facecolor=colors[i], alpha=0.7, boxstyle="round"),
                #     )

        # Visualize initial chamber
        if show_initial_chamber and initial_chamber is not None:
            x, y, r = initial_chamber
            # x = x / orig_w * w
            # y = y / orig_h * h
            # r = 0.5 * r / orig_w * w
            circle = Circle((x, y), 0.5 * r, fill=False, edgecolor="red", linewidth=2)
            ax.add_patch(circle)

        for idx, chomata_pos in chomata_result.items():
            if len(chomata_pos) > 1:  # successfully detect 2 chomatas in a voludion
                cv2.circle(binary_img_rgb, chomata_pos[0][:2], 2, (0, 0, 255), 2)
                cv2.circle(binary_img_rgb, chomata_pos[1][:2], 2, (0, 0, 255), 2)

        # Add title and legend
        ax.set_title(f"Volution Analysis: {img_name} (Binarized)", fontsize=14)
        if show_volution_lines:
            ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

        # Remove axes
        ax.axis("off")

        # Save binarized figure with markings
        binary_output_path = os.path.join(output_dir, f"{img_name}_binary.{save_format}")
        plt.tight_layout()
        plt.savefig(binary_output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Visualizations saved to {original_output_path} and {binary_output_path}")


def batch_visualize(
    input_dir,
    output_dir,
    feat_recog_args,
    file_extensions=(".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    **kwargs,
):
    """
    Process all images in a directory and visualize volutions.

    Parameters:
    input_dir (str): Directory containing input images
    output_dir (str): Directory to save visualization results
    feat_recog_args: Arguments for VolutionCounter
    file_extensions (tuple): File extensions to process
    **kwargs: Additional arguments for visualize_volutions
    """
    # Get all image files in the input directory
    img_paths = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(file_extensions):
            img_paths.append(os.path.join(input_dir, filename))

    if not img_paths:
        print(f"No images found in {input_dir} with extensions {file_extensions}")
        return

    print(f"Found {len(img_paths)} images to process")
    visualize_volutions(img_paths, output_dir, feat_recog_args, **kwargs)
    print(f"Visualization complete. Results saved to {output_dir}")


def main():
    img_path_root = "dataset/common/visualize_test"
    img_paths = os.listdir(img_path_root)
    # img_paths = [f"{img_path_root}/{img_path}" for img_path in img_paths]
    img_paths = ["dataset/common/images/Fusulina_knichti_1_2.png"]
    # test_image_path = "dataset/instructions_no_vis_tools/instructions_test.jsonl"
    # test_images = []
    # with open(test_image_path, "r") as f:
    #     for line in f:
    #         img_path = json.loads(line)["image"]
    #         test_images.append(f"dataset/common/images/{img_path}")
    output_dir = "./visualize_tools"
    visualize_volutions(img_paths, output_dir, feat_recog_args)


if __name__ == "__main__":
    main()

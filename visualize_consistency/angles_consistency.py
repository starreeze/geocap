import json
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if __name__ == "__main__":
    with open("dataset/common/extracted_info/new_testset_reference_info.json", "r") as f:
        reference_info = json.load(f)

    # Map species name to equator and poles
    img2equator = {}
    img2poles = {}
    for reference_info in reference_info:
        image_path = reference_info["image_path"]
        img2equator[image_path] = reference_info["equator"]
        img2poles[image_path] = reference_info["poles"]

    with open("dataset/common/extracted_info/angles_info.json", "r") as f:
        angles_info_list = json.load(f)

    # equator classification
    convex_equator_list = []
    concave_equator_list = []
    straight_equator_list = []
    # poles classification
    blunted_poles_list = []
    pointed_poles_list = []
    elongated_poles_list = []
    for angles_info in angles_info_list:
        image_path = angles_info["image_path"]
        if image_path not in img2equator:
            continue

        # Equator
        equator_angle = (angles_info["upper_angle"] + angles_info["lower_angle"]) / 2
        equator = img2equator[image_path]
        # ignore unclear description
        if ("concave" in equator or "depress" in equator) and (
            "convex" in equator or "inflated" in equator or "stout" in equator or "bulg" in equator
        ):
            continue
        elif "concave" in equator or "depress" in equator:
            concave_equator_list.append((equator_angle, image_path))
        elif (
            "convex" in equator or "inflated" in equator or "stout" in equator or "bulg" in equator
        ):  # bulge or bulging
            convex_equator_list.append((equator_angle, image_path))
        else:
            straight_equator_list.append((equator_angle, image_path))

        # Poles
        poles_angle = (angles_info["left_angle"] + angles_info["right_angle"]) / 2
        poles = img2poles[image_path]
        # ignore unclear description
        if ("round" in poles and "point" in poles) or ("blunt" in poles and "sharp" in poles):
            continue

        # Handle adv. first
        adv_pattern = re.compile(r"\b\w+ly\b")
        advs_in_poles = adv_pattern.findall(poles)
        found_adv = False
        for adv in advs_in_poles:
            if adv in ["bluntly", "roundly", "obtusely"]:
                found_adv = True
                blunted_poles_list.append((poles_angle, image_path))
            elif adv in ["sharply", "narrowly"]:
                found_adv = True
                pointed_poles_list.append((poles_angle, image_path))

        if not found_adv:
            if "blunt" in poles or "round" in poles:
                blunted_poles_list.append((poles_angle, image_path))
            elif "point" in poles or "sharp" in poles or "taper" in poles:
                pointed_poles_list.append((poles_angle, image_path))
            elif "elongate" in poles or "extended" in poles:
                elongated_poles_list.append((poles_angle, image_path))

    print(
        f"Total samples: {len(convex_equator_list) + len(concave_equator_list) + len(straight_equator_list)}"
    )
    print(f"Convex samples: {len(convex_equator_list)}")
    print(f"Concave samples: {len(concave_equator_list)}")
    print(f"Straight samples: {len(straight_equator_list)}")

    # 输出每类的极值以及对应的物种名
    # Convex equator top 5 min and max
    convex_sorted = sorted(convex_equator_list)
    print(f"Convex equator top 5 min: {convex_sorted[:5]}")
    print(f"Convex equator top 5 max: {convex_sorted[-5:]}")

    # Concave equator top 5 min and max
    concave_sorted = sorted(concave_equator_list)
    print(f"Concave equator top 5 min: {concave_sorted[:5]}")
    print(f"Concave equator top 5 max: {concave_sorted[-5:]}")

    # Straight equator top 5 min and max
    straight_sorted = sorted(straight_equator_list)
    print(f"Straight equator top 5 min: {straight_sorted[:5]}")
    print(f"Straight equator top 5 max: {straight_sorted[-5:]}")

    print(f"\nBlunted samples: {len(blunted_poles_list)}")
    print(f"Pointed samples: {len(pointed_poles_list)}")
    print(f"Elongated samples: {len(elongated_poles_list)}")

    # 输出poles每类的极值以及对应的物种名
    # Blunted poles top 5 min and max
    blunted_sorted = sorted(blunted_poles_list)
    print(f"Blunted poles top 5 min: {blunted_sorted[:5]}")
    print(f"Blunted poles top 5 max: {blunted_sorted[-5:]}")

    # Pointed poles top 5 min and max
    pointed_sorted = sorted(pointed_poles_list)
    print(f"Pointed poles top 5 min: {pointed_sorted[:5]}")
    print(f"Pointed poles top 5 max: {pointed_sorted[-5:]}")

    # Elongated poles top 5 min and max
    elongated_sorted = sorted(elongated_poles_list)
    print(f"Elongated poles top 5 min: {elongated_sorted[:5]}")
    print(f"Elongated poles top 5 max: {elongated_sorted[-5:]}")

    # Visualize equator distribution as stacked histogram
    plt.figure(figsize=(10, 6))

    # Extract angle values for plotting
    convex_angles = [angle for angle, _ in convex_equator_list]
    concave_angles = [angle for angle, _ in concave_equator_list]
    straight_angles = [angle for angle, _ in straight_equator_list]

    # Define bins to cover the range of all data
    all_angles = convex_angles + concave_angles + straight_angles
    bins = np.linspace(min(all_angles), max(all_angles), 20)

    # Create stacked histogram
    plt.hist(
        [convex_angles, concave_angles, straight_angles],
        bins=bins,
        label=[
            f"Convex (n={len(convex_angles)})",
            f"Concave (n={len(concave_angles)})",
            f"Straight (n={len(straight_angles)})",
        ],
        color=["blue", "green", "red"],
        alpha=0.8,
        stacked=True,
        edgecolor="black",
        linewidth=0.5,
        rwidth=0.8,
    )

    plt.xlabel("Angle (rad)")
    plt.ylabel("Count")
    plt.title("Equator Angle Distribution by Shape Category")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()
    plt.savefig("visualize_consistency/results/equator_distribution.png")

    # Visualize equator distribution as density curves
    plt.figure(figsize=(10, 6))

    # Create density curves for equator
    equator_data = [convex_angles, concave_angles, straight_angles]
    equator_labels = ["Convex", "Concave", "Straight"]
    equator_colors = ["blue", "green", "red"]

    x_min, x_max = min(all_angles), max(all_angles)
    x_range = np.linspace(x_min, x_max, 200)

    # Calculate total number of samples for normalization
    total_samples = len(all_angles)

    for data, label, color in zip(equator_data, equator_labels, equator_colors):
        if len(data) > 1:
            # Use Gaussian KDE for density estimation
            kde = stats.gaussian_kde(data)
            density = kde(x_range)

            # Scale density by the proportion of this category in total samples
            scaled_density = density * len(data) / total_samples

            # Plot the curve
            plt.plot(x_range, scaled_density, color=color, linewidth=2, label=f"{label} (n={len(data)})")

            # Fill under the curve with transparency
            plt.fill_between(x_range, scaled_density, alpha=0.3, color=color)

    plt.xlabel("Angle (rad)")
    plt.ylabel("Density")
    plt.title("Equator Angle Distribution Density Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualize_consistency/results/equator_density_curves.png")

    # Visualize poles distribution as stacked histogram
    plt.figure(figsize=(10, 6))

    # Extract angle values for plotting
    blunted_angles = [angle for angle, _ in blunted_poles_list]
    pointed_angles = [angle for angle, _ in pointed_poles_list]
    elongated_angles = [angle for angle, _ in elongated_poles_list]

    # Define bins to cover the range of all data
    all_angles = blunted_angles + pointed_angles + elongated_angles
    bins = np.linspace(min(all_angles), max(all_angles), 20)

    # Create stacked histogram
    plt.hist(
        [blunted_angles, pointed_angles, elongated_angles],
        bins=bins,
        label=[
            f"Blunted (n={len(blunted_angles)})",
            f"Pointed (n={len(pointed_angles)})",
            f"Elongated (n={len(elongated_angles)})",
        ],
        color=["blue", "green", "red"],
        alpha=0.8,
        stacked=True,
        edgecolor="black",
        linewidth=0.5,
        rwidth=0.8,
    )

    plt.xlabel("Angle (rad)")
    plt.ylabel("Count")
    plt.title("Poles Angle Distribution by Shape Category")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()
    plt.savefig("visualize_consistency/results/poles_distribution.png")

    # Visualize poles distribution as density curves
    plt.figure(figsize=(10, 6))

    # Create density curves for poles
    poles_data = [blunted_angles, pointed_angles, elongated_angles]
    poles_labels = ["Blunted", "Pointed", "Elongated"]
    poles_colors = ["blue", "green", "red"]

    x_min, x_max = min(all_angles), max(all_angles)
    x_range = np.linspace(x_min, x_max, 200)

    # Calculate total number of samples for normalization
    total_samples = len(all_angles)

    for data, label, color in zip(poles_data, poles_labels, poles_colors):
        if len(data) > 1:
            # Use Gaussian KDE for density estimation
            kde = stats.gaussian_kde(data)
            density = kde(x_range)

            # Scale density by the proportion of this category in total samples
            scaled_density = density * len(data) / total_samples

            # Plot the curve
            plt.plot(x_range, scaled_density, color=color, linewidth=2, label=f"{label} (n={len(data)})")

            # Fill under the curve with transparency
            plt.fill_between(x_range, scaled_density, alpha=0.3, color=color)

    plt.xlabel("Angle (rad)")
    plt.ylabel("Density")
    plt.title("Poles Angle Distribution Density Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualize_consistency/results/poles_density_curves.png")

    plt.show()

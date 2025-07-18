import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if __name__ == "__main__":
    with open("dataset/common/extracted_info/new_testset_reference_info.json", "r") as f:
        reference_info = json.load(f)

    # Map species name to slope
    img2slope = {}
    for reference_info in reference_info:
        image_path = reference_info["image_path"]
        img2slope[image_path] = reference_info["lateral_slopes"]

    with open("dataset/common/extracted_info/slopes_info.json", "r") as f:
        slopes_info_list = json.load(f)

    # slope classification
    convex_slope_list = []
    concave_slope_list = []
    straight_slope_list = []

    for slopes_info in slopes_info_list:
        image_path = slopes_info["image_path"]
        if image_path not in img2slope:
            continue

        # slope
        slope_score = 0
        for score in slopes_info["convex_scores"]:
            slope_score += score
        if abs(slope_score) > 10:
            print(f"Slope score: {slope_score} for {image_path}")
            continue
        slope_desc = img2slope[image_path]
        # ignore unclear description
        if ("concave" in slope_desc or "depress" in slope_desc) and ("convex" in slope_desc):
            continue
        elif "concave" in slope_desc or "depress" in slope_desc:
            concave_slope_list.append((slope_score, image_path))
        elif "convex" in slope_desc:
            convex_slope_list.append((slope_score, image_path))
        else:
            straight_slope_list.append((slope_score, image_path))

    print(f"Total samples: {len(convex_slope_list) + len(concave_slope_list) + len(straight_slope_list)}")
    print(f"Convex samples: {len(convex_slope_list)}")
    print(f"Concave samples: {len(concave_slope_list)}")
    print(f"Straight samples: {len(straight_slope_list)}")

    convex_scores = [score for score, _ in convex_slope_list]
    concave_scores = [score for score, _ in concave_slope_list]
    straight_scores = [score for score, _ in straight_slope_list]

    # Visualize equator distribution as stacked histogram
    plt.figure(figsize=(10, 6))

    # Define bins to cover the range of all data
    all_scores = convex_scores + concave_scores + straight_scores
    bins = np.linspace(min(all_scores), max(all_scores), 20)

    # Create stacked histogram
    plt.hist(
        [convex_scores, concave_scores, straight_scores],
        bins=bins,
        label=[
            f"Convex (n={len(convex_scores)})",
            f"Concave (n={len(concave_scores)})",
            f"Straight (n={len(straight_scores)})",
        ],
        color=["blue", "green", "red"],
        alpha=0.8,
        stacked=True,
        edgecolor="black",
        linewidth=0.5,
        rwidth=0.8,
    )

    plt.xlabel("Slope Score")
    plt.ylabel("Count")
    plt.title("Slope Score Distribution by Shape Category")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()
    plt.savefig("visualize_consistency/results/slope_distribution.png")

    # Visualize equator distribution as density curves
    plt.figure(figsize=(10, 6))

    # Create density curves for equator
    slope_data = [convex_scores, concave_scores, straight_scores]
    slope_labels = ["Convex", "Concave", "Straight"]
    slope_colors = ["blue", "green", "red"]

    x_min, x_max = min(all_scores), max(all_scores)
    x_range = np.linspace(x_min, x_max, 200)

    total_samples = len(all_scores)

    for data, label, color in zip(slope_data, slope_labels, slope_colors):
        if len(data) > 1:
            # Use Gaussian KDE for density estimation
            kde = stats.gaussian_kde(data)
            density = kde(x_range)

            scaled_density = density * len(data) / total_samples

            # Plot the curve
            plt.plot(x_range, scaled_density, color=color, linewidth=2, label=f"{label} (n={len(data)})")

            # Fill under the curve with transparency
            plt.fill_between(x_range, scaled_density, alpha=0.3, color=color)

    convex_sorted = sorted(convex_slope_list)
    print(f"Convex top 5 min: {convex_sorted[:5]}")
    print(f"Convex top 5 max: {convex_sorted[-5:]}")

    # Concave poles top 5 min and max
    concave_sorted = sorted(concave_slope_list)
    print(f"Concave top 5 min: {concave_sorted[:5]}")
    print(f"Concave top 5 max: {concave_sorted[-5:]}")

    # Elongated poles top 5 min and max
    straight_sorted = sorted(straight_slope_list)
    print(f"Straight top 5 min: {straight_sorted[:5]}")
    print(f"Straight top 5 max: {straight_sorted[-5:]}")

    plt.xlabel("Slope Score")
    plt.ylabel("Density")
    plt.title("Slope Score Distribution Density Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualize_consistency/results/slope_density_curves.png")
    plt.show()

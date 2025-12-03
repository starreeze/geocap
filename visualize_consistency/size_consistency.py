import json

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

if __name__ == "__main__":
    with open("dataset/common/extracted_info/species_desc.json", "r") as f:
        reference_info = json.load(f)

    # Map species name to size
    species2size = {}
    for reference_info in reference_info:
        species_name = reference_info["species_name"]
        species2size[species_name] = reference_info["size"]

    with open("dataset/common/extracted_info/size_info.json", "r") as f:
        size_info = json.load(f)

    # Length statistics
    small_length_list = []
    medium_length_list = []
    large_length_list = []
    for size_info in size_info:
        image_path = size_info["image_path"]
        species_name = "_".join(image_path.split("_")[:3])
        if species_name not in species2size:
            continue
        size = species2size[species_name]
        if "small" in size:
            small_length_list.append(size_info["length"])
        elif "large" in size:
            large_length_list.append(size_info["length"])
        elif "medium" in size or "moderate" in size:
            medium_length_list.append(size_info["length"])

    print(f"Total samples: {len(small_length_list) + len(medium_length_list) + len(large_length_list)}")
    print(f"Small samples: {len(small_length_list)}")
    print(f"Medium samples: {len(medium_length_list)}")
    print(f"Large samples: {len(large_length_list)}")

    # Visualize length distribution as stacked histogram
    plt.figure(figsize=(10, 6))

    # Define bins to cover the range of all data
    all_lengths = small_length_list + medium_length_list + large_length_list
    bins = np.linspace(min(all_lengths), max(all_lengths), 20)

    # Create stacked histogram
    plt.hist(
        [small_length_list, medium_length_list, large_length_list],
        bins=bins,
        label=[
            f"Small (n={len(small_length_list)})",
            f"Medium (n={len(medium_length_list)})",
            f"Large (n={len(large_length_list)})",
        ],
        color=["blue", "green", "red"],
        alpha=0.8,
        stacked=True,
        edgecolor="black",
        linewidth=0.5,
        rwidth=0.8,
    )

    plt.xlabel("Length (mm)")
    plt.ylabel("Count")
    plt.title("Length Distribution by Size Category")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Improve layout
    plt.tight_layout()
    plt.savefig("visualize_consistency/results/size_distribution.png")

    # Visualize poles distribution as density curves
    plt.figure(figsize=(10, 6))

    # Create density curves for poles
    size_data = [small_length_list, medium_length_list, large_length_list]
    size_labels = ["Small", "Medium", "Large"]
    size_colors = ["blue", "green", "red"]

    x_min, x_max = min(all_lengths), max(all_lengths)
    x_range = np.linspace(x_min, x_max, 200)

    # Calculate total number of samples for normalization
    total_samples = len(all_lengths)

    for data, label, color in zip(size_data, size_labels, size_colors):
        if len(data) > 0:
            # Use Gaussian KDE for density estimation
            kde = stats.gaussian_kde(data)
            density = kde(x_range)

            # Scale density by the proportion of this category in total samples
            scaled_density = density * len(data) / total_samples

            # Plot the curve
            plt.plot(x_range, scaled_density, color=color, linewidth=2, label=f"{label} (n={len(data)})")

            # Fill under the curve with transparency
            plt.fill_between(x_range, scaled_density, alpha=0.3, color=color)

    plt.xlabel("Length (mm)")
    plt.ylabel("Density")
    plt.title("Size Distribution Density Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("visualize_consistency/results/size_density_curves.png")
    plt.show()

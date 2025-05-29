import csv
import json
from collections import defaultdict

from common.args import fossil_eval_args

characteristics = [
    "overall_size",
    "overall_shape",
    "length",
    "width",
    "ratio",
    "axis_shape",
    "number_of_volutions",
    "thickness_of_spirotheca",
    "height_of_volution",
    "septa_folds",
    "proloculus",
    "tunnel_shape",
    "tunnel_angles",
    "chomata",
    "axial_filling",
]


def feature_statistics():
    with open(f"{fossil_eval_args.eval_result_dir}/detailed_score_list.json", "r") as f:
        data = json.load(f)

    slice_tab = {char: {"rating": 0.0, "valid_count": 0} for char in characteristics}

    for entry in data:
        for char in characteristics:
            if char not in entry or entry[char]["rating"] == -1:
                continue
            slice_tab[char]["rating"] += entry[char]["rating"]
            slice_tab[char]["valid_count"] += 1

    average_score = 0
    for char in characteristics:
        slice_tab[char]["rating"] = round(slice_tab[char]["rating"] / slice_tab[char]["valid_count"], 2)
        average_score += slice_tab[char]["rating"]

    # Write to CSV instead of JSONL
    with open(f"{fossil_eval_args.eval_result_dir}/feature_statistics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Rating", "Valid Count"])
        writer.writerow(["Average", round(average_score / len(characteristics), 2), ""])
        for char in characteristics:
            writer.writerow([char, slice_tab[char]["rating"], slice_tab[char]["valid_count"]])


def species_statistics():
    """
    Calculate statistics for each species based on image path prefixes.
    """
    with open(f"{fossil_eval_args.eval_result_dir}/detailed_score_list.json", "r") as f:
        data = json.load(f)

    # Group entries by species
    species_data = defaultdict(list)
    for entry in data:
        species_name = "_".join(entry["image_path"].split("_")[:2])
        species_data[species_name].append(entry)

    # Calculate statistics for each species
    species_characteristics = {char: [] for char in characteristics}

    i = 0
    for species, entries in species_data.items():
        species_slice_tab = {char: {"rating": 0.0, "valid_count": 0} for char in characteristics}

        for entry in entries:
            i += 1

            for char in characteristics:
                if char not in entry or entry[char]["rating"] == -1:
                    continue
                species_slice_tab[char]["rating"] += entry[char]["rating"]
                species_slice_tab[char]["valid_count"] += 1

        # Calculate average for each characteristic for this species
        for char in characteristics:
            if species_slice_tab[char]["valid_count"] > 0:
                avg_rating = round(
                    species_slice_tab[char]["rating"] / species_slice_tab[char]["valid_count"], 2
                )
                species_characteristics[char].append(avg_rating)

    # Calculate average rating across all species for each characteristic
    avg_rating_over_species = {}
    for char in characteristics:
        if species_characteristics[char]:  # Only calculate if there are valid ratings
            avg_rating_over_species[char] = round(
                sum(species_characteristics[char]) / len(species_characteristics[char]), 2
            )
        else:
            avg_rating_over_species[char] = 0.0

    # Calculate overall average across all characteristics
    valid_characteristics = [char for char in characteristics if species_characteristics[char]]
    if valid_characteristics:
        average_score = sum(avg_rating_over_species[char] for char in valid_characteristics) / len(
            valid_characteristics
        )
    else:
        average_score = 0.0

    # Write species statistics to CSV instead of JSONL
    with open(f"{fossil_eval_args.eval_result_dir}/species_statistics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Feature", "Rating", "Species Count"])
        writer.writerow(["Average", round(average_score, 2), len(species_data)])
        for char in characteristics:
            writer.writerow([char, avg_rating_over_species[char], len(species_characteristics[char])])


def statistics():
    feature_statistics()
    species_statistics()


def main():
    statistics()


if __name__ == "__main__":
    main()

from common.args import fossil_eval_args
import os
import json
from tqdm import tqdm


def main():
    with open(f"{fossil_eval_args.eval_result_dir}/detailed_score_list.json", "r") as f:
        data = json.load(f)
    scoreboard = []
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
        "proloculus",
        "tunnel_shape",
        "tunnel_angles",
        "chomata",
        "axial_filling",
    ]
    slice_tab = {char: 0 for char in characteristics}
    total_avg_score = 0
    for entry in tqdm(data):
        ratings = []
        for char in characteristics:
            if char not in entry:
                continue
            slice_tab[char] += entry[char]["rating"]
            ratings.append(entry[char]["rating"])
        avg = (sum(ratings)) / len(characteristics)
        maxx = max(ratings)
        minn = min(ratings)
        scoreboard.append({"average": avg, "max": maxx, "min": minn})
        total_avg_score += avg
    with open(f"{fossil_eval_args.eval_result_dir}/statistics.jsonl", "w") as f:
        f.write(f'[\n{{"Average":{total_avg_score / len(scoreboard):.2f}}},\n')
        for char in characteristics:
            f.write(f'{{"{char}": {slice_tab[char] / len(scoreboard):.2f}}},\n')
        f.write("]\n")
        for sc in scoreboard:
            f.write(json.dumps(sc))
            f.write("\n")


if __name__ == "__main__":
    main()

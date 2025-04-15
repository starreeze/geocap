"construct descriptions according to generated rules"

import hashlib
import json
import math
import os
import random
import re
from typing import Any

from tqdm import tqdm

from common.args import caption_args, data_args, run_args
from data.caption.caption_2nd.chomata import Chomata
from data.caption.caption_2nd.deposit import Deposit
from data.caption.caption_2nd.proloculus import Proloculus
from data.caption.caption_2nd.septa import Septa
from data.caption.caption_2nd.shell import Shell
from data.caption.caption_2nd.tunnel import Tunnel
from data.caption.caption_2nd.volution import Volution
from data.caption.prompt import *


def caption(rules: list[dict[str, Any]], generator, output_path: str):
    # TODO generate captions. Use whatever technique you like, e.g., demonstrations, cot, ...
    # TODO also consider how to handle distance:
    #      It should be mentioned in input prompt. Try scaling to produce different distances.
    # NOTE see common/llm.py for the use of open-source LLMs.
    # test='''[{"type": "line", "points": [[0, 0], [3, 3]]},{"type": "polygon", "points": [[1, 1], [2, 1], [1, 2], [2, 2]]},{"type": "polygon", "points": [[0, 10], [-5, 5], [5, 5]]}]'''
    # rule_str=json.dumps(rules)
    # rules=rules[100:200]
    rule_strs = []
    input_texts: list[str] = []
    idx = 0
    for rule in tqdm(rules):
        input_str, rule_str = gen_user_input_txt_2nd(rule)
        rule_strs.append({"input": input_str, "output": rule_str})
        idx += 1

    with open(output_path, "w") as f:
        f.write(json.dumps(rule_strs, indent=4))
        f.flush()


def gen_user_input_txt_2nd(rule):
    txt = ""
    obj_parts = []

    def get_volution_index(shape):
        a = shape["special_info"]
        return int(a[len("volution ") :])

    volution_max = {}
    initial_chamber = {}
    chomata_shapes = []
    volutions = []
    for shape in rule["shapes"]:
        if re.match("volution [0-9]+", shape["special_info"]) is not None:
            if shape["type"] == "ellipse":
                shape["vertices"] = []
            volutions.append(shape)
            if volution_max == {}:
                volution_max = shape
            else:
                if get_volution_index(shape) > get_volution_index(volution_max):
                    volution_max = shape
        elif re.match("initial chamber", shape["special_info"]) is not None:
            initial_chamber = shape
        elif re.match("chomata of volution [0-9]+", shape["special_info"]) is not None:
            chomata_shapes.append(shape)
    if volution_max["type"] == "ellipse":
        volution_max["ratio"] = volution_max["major_axis"] / volution_max["minor_axis"]
        volution_max["width"] = volution_max["major_axis"]
        volution_max["height"] = volution_max["minor_axis"]
        volution_max["vertices"] = []
        volution_max["control_points"] = []
    elif volution_max["type"] == "curves":
        volution_max["width"] = abs(volution_max["vertices"][2][0] - volution_max["vertices"][0][0])
        volution_max["height"] = abs(volution_max["vertices"][1][1] - volution_max["vertices"][3][1])
    elif "fusiform" in volution_max["type"]:
        volution_max["width"] = volution_max["x_end"] - volution_max["x_start"]
        volution_max["height"] = volution_max["width"] / volution_max["ratio"]
        volution_max["vertices"] = []
        volution_max["control_points"] = []
    obj_parts.append(
        Shell(
            volution_max["type"],
            volution_max["ratio"],
            volution_max["width"],
            volution_max["height"],
            volution_max["control_points"],
            volution_max["vertices"],
            initial_chamber["center"],
        )
    )
    obj_parts.append(
        Volution(
            rule["numerical_info"]["num_volutions"],
            [
                ["{:.1f} mm".format(random.random()), i]
                for i in range(int(rule["numerical_info"]["num_volutions"]))
            ],
            volutions,
        )
    )

    obj_parts.append(
        Proloculus("", initial_chamber, (initial_chamber["major_axis"] + initial_chamber["minor_axis"]) / 2)
    )
    # if len(chomata_shapes) > 0:
    obj_parts.append(Chomata(chomata_shapes, rule["numerical_info"]["num_volutions"], volutions))
    if "tunnel_angles" in rule["numerical_info"] and len(rule["numerical_info"]["tunnel_angles"]) > 0:
        obj_parts.append(
            Tunnel(
                rule,
                rule["numerical_info"]["visible_chomata_idx"],
                obj_parts[-1].chomata_whs_relative,
                rule["numerical_info"]["tunnel_angles"],
            )
        )
    if "axial_filling" in rule and len(rule["axial_filling"]) > 0:
        obj_parts.append(Deposit(rule["axial_filling"], rule["numerical_info"]["num_volutions"]))
    else:
        obj_parts.append(Deposit([], rule["numerical_info"]["num_volutions"]))
    obj_parts.append(Septa(rule["septa_folds"]))
    txt2 = head_start_2nd + "\n"
    feature_tagged = []
    for part in obj_parts:
        feature_tagged.extend(part.genUserInput())
        txt2 += part.genInput() + ""
    feature_tagged.sort(key=featureSortFunc)
    for feat in feature_tagged:
        txt += feat
    return txt2.strip(), txt.strip()


def featureSortFunc(feat):
    feat_order = [
        "shell",
        "length",
        "width",
        "ratio",
        "volution",
        "proloculus",
        "axis",
        "axial filling",
        "spirotheca",
        "septa",
        "chomata",
        "tunnel shape",
        "tunnel angle",
    ]
    match = re.search(r"<(.*?)>", feat)
    if match:
        return feat_order.index(match.group(1))
    else:
        raise ValueError(f"feature failed: {feat}")


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)[run_args.start_pos : run_args.end_pos]
    caption(samples, None, data_args.caption_path)


if __name__ == "__main__":
    main()

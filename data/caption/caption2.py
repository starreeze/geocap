"construct descriptions according to generated rules"

import hashlib
import json
import math
import os
import random
from typing import Any

from data.caption_2nd.shell import Shell
from data.caption_2nd.volution import Volution
from data.caption_2nd.tunnel import Tunnel
from data.caption_2nd.proloculus import Proloculus
from data.caption_2nd.chomata import Chomata
import re
from common.args import caption_args, data_args, run_args


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
    for rule in rules:
        rule_str = gen_user_input_txt_2nd(rule)
        rule_strs.append(rule_str)
        idx += 1
        print("finished:", str(idx) + "/" + str(len(rules)))

    with open(output_path, "w") as f:
        f.write(json.dumps(rule_strs, indent=4))
        f.flush()


def gen_user_input_txt_2nd(rule):
    txt = ""
    obj_parts = []

    def get_volution_index(shape):
        a = shape["special_info"].split(".")[0]
        return int(a[len("volution ") :])

    volution_max = {}
    initial_chamber = {}
    chomata_shapes = []
    for shape in rule["shapes"]:
        if re.match("volution [0-9]+\\.", shape["special_info"]) is not None:
            if volution_max == {}:
                volution_max = shape
            else:
                if get_volution_index(shape) > get_volution_index(volution_max):
                    volution_max = shape
        elif re.match("initial chamber\\. ", shape["special_info"]) is not None:
            initial_chamber = shape
        elif re.match("chomata of volution [0-9]+\\. ", shape["special_info"]) is not None:
            chomata_shapes.append(shape)
    if volution_max["type"] == "ellipse":
        volution_max["ratio"] = volution_max["major_axis"] / volution_max["minor_axis"]
        volution_max["width"] = volution_max["major_axis"]
        volution_max["height"] = volution_max["minor_axis"]
        volution_max["control_points"] = []
    elif volution_max["type"] == "curves":
        volution_max["width"] = volution_max["vertices"][2][0] - volution_max["vertices"][0][0]
        volution_max["height"] = volution_max["vertices"][1][1] - volution_max["vertices"][3][1]
    obj_parts.append(
        Shell(
            volution_max["type"],
            volution_max["ratio"],
            volution_max["width"],
            volution_max["height"],
            volution_max["control_points"],
            volution_max["vertices"],
        )
    )
    obj_parts.append(
        Volution(
            rule["numerical_info"]["num_volutions"],
            [["{:.1f} mm".format(random.random()), i] for i in range(int(rule["numerical_info"]["num_volutions"]))],
        )
    )
    obj_parts.append(Tunnel(rule["numerical_info"]["tunnel_start_idx"], rule["numerical_info"]["tunnel_angles"]))
    obj_parts.append(
        Proloculus("", initial_chamber, (initial_chamber["major_axis"] + initial_chamber["minor_axis"]) / 2)
    )
    obj_parts.append(Chomata(chomata_shapes, rule["numerical_info"]["num_volutions"]))
    for part in obj_parts:
        txt += part.genUserInput() + ""
    return txt


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
    caption(samples, None, "dataset/caption_2nd_WIP.json")


if __name__ == "__main__":
    main()

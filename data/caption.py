"construct descriptions according to generated rules"

import hashlib
import json
import math
import os
import random
from typing import Any

from scipy import special
from tqdm import tqdm

from common.args import caption_args, data_args, run_args
from common.llm import LLMGenerator, generator_mapping, model_path_mapping
from common.prompt import *


def caption(rules: list[dict[str, Any]], generator: LLMGenerator, output_path: str):
    # TODO generate captions. Use whatever technique you like, e.g., demonstrations, cot, ...
    # TODO also consider how to handle distance:
    #      It should be mentioned in input prompt. Try scaling to produce different distances.
    # NOTE see common/llm.py for the use of open-source LLMs.
    # test='''[{"type": "line", "points": [[0, 0], [3, 3]]},{"type": "polygon", "points": [[1, 1], [2, 1], [1, 2], [2, 2]]},{"type": "polygon", "points": [[0, 10], [-5, 5], [5, 5]]}]'''
    # rule_str=json.dumps(rules)
    rule_strs: list[str] = []
    input_texts: list[str] = []
    for rule in rules:
        rule_str, gen_input_text = gen_user_input(rule)
        rule_strs.append(rule_str)
        input_texts.append(gen_input_text)

    messages = [
        [{"role": "system", "content": context}, {"role": "user", "content": rule_strs[i]}]
        for i in range(len(rule_strs))
    ]
    bs = caption_args.caption_batchsize
    output_texts = generator(messages, bs)
    total_size = (len(input_texts) + bs - 1) // bs
    with open(output_path, "w") as f:
        for batch_idx, outputs in tqdm(enumerate(output_texts), total=total_size):
            inputs = input_texts[batch_idx * bs : (batch_idx + 1) * bs]
            for input, output in zip(inputs, outputs):
                f.write(json.dumps({"input": input, "output": output}) + "\n")
                f.flush()


def euc_dist(p1, p2):
    return math.sqrt(abs((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def rect_width_height(rect):
    dists = [euc_dist(rect[0], rect[1]), euc_dist(rect[0], rect[2]), euc_dist(rect[0], rect[3])]
    dists.sort()
    return dists[1], dists[0]


def points_center(points):
    x = 0
    y = 0
    for point in points:
        x += point[0]
        y += point[1]
    return [x / len(points), y / len(points)]


def calculate_polygon_C(coords):
    C = 0
    for i in range(len(coords)):
        C += euc_dist(coords[i - 1], coords[i])
    return C


def calculate_ellipse_C(a, b):
    e_sq = 1.0 - b**2 / a**2
    C = 4 * a * special.ellipe(e_sq)
    return C


def calculate_spiral_C(initial_radius, growth_rate, max_theta, n_sample):
    a = initial_radius
    b = growth_rate
    points = []
    start_theta = max_theta - 2 * math.pi
    for i in range(n_sample):
        theta = start_theta + 2 * math.pi * i / n_sample
        r = a + b * theta
        points.append([r * math.cos(theta), r * math.sin(theta)])
    return calculate_polygon_C(points)


def get_center_and_weight(shape, n_digits=4):
    if shape["type"] in ["segment", "polygon"]:
        p_center = points_center(shape["points"])
        return [round(p_center[0], ndigits=n_digits), round(p_center[1], ndigits=n_digits)], calculate_polygon_C(
            shape["points"]
        )
    elif shape["type"] in ["spiral"]:
        return [
            round(shape["center"][0], ndigits=n_digits),
            round(shape["center"][1], ndigits=n_digits),
        ], calculate_spiral_C(shape["initial_radius"], shape["growth_rate"], shape["max_theta"], 360)
    elif shape["type"] in ["ellipse"]:
        return [
            round(shape["center"][0], ndigits=n_digits),
            round(shape["center"][1], ndigits=n_digits),
        ], calculate_ellipse_C(shape["major_axis"], shape["minor_axis"])
    elif shape["type"] in ["segment", "line", "ray"]:
        p_center = points_center(shape["points"])
        return [round(p_center[0], ndigits=n_digits), round(p_center[1], ndigits=n_digits)], euc_dist(
            shape["points"][0], shape["points"][1]
        )
    return [0, 0], 0


def decide_abs_pos(center_positions, x1, x2, y1, y2):
    abs_positions = []
    for center in center_positions:
        x = center[0]
        y = center[1]
        if x < x1:
            if y < y1:
                abs_positions.append("bottom_left")
            elif y <= y2:
                abs_positions.append("left")
            else:
                abs_positions.append("top_left")
        elif x <= x2:
            if y < y1:
                abs_positions.append("bottom")
            elif y <= y2:
                abs_positions.append("center")
            else:
                abs_positions.append("top")
        else:
            if y < y1:
                abs_positions.append("bottom_right")
            elif y <= y2:
                abs_positions.append("right")
            else:
                abs_positions.append("top_right")
    return abs_positions


def gen_grid(shapes):
    center_positions = []
    weights = []
    mass_center = [0.0, 0.0]
    for shape in shapes:
        center_pos, weight = get_center_and_weight(shape)
        center_positions.append(center_pos)
        weights.append(weight)
        mass_center[0] += center_pos[0] * weight
        mass_center[1] += center_pos[1] * weight
    sum_weights = sum(weights)
    mass_center[0] /= sum_weights
    mass_center[1] /= sum_weights
    points_x = [point[0] for point in center_positions]
    points_y = [point[1] for point in center_positions]
    edge_left = min(points_x)
    edge_right = max(points_x)
    edge_top = max(points_y)
    edge_bottom = min(points_y)
    x1 = (edge_left + mass_center[0]) / 2
    x2 = (edge_right + mass_center[0]) / 2
    y1 = (edge_bottom + mass_center[1]) / 2
    y2 = (edge_top + mass_center[1]) / 2
    # [0,1] range canvas
    # x1=1/3
    # x2=2/3
    # y1=1/3
    # y2=2/3
    return center_positions, decide_abs_pos(center_positions, x1, x2, y1, y2)


def gen_position_description(shapes):
    return gen_grid(shapes)


def gen_user_input(rules):
    ret_text = "TOTAL SHAPES: {}\n".format(len(rules["shapes"]))
    ret_text += "BEGIN SHAPES\n"
    polygon_special = [0, 0, 0, "triangle", "quadrilateral", "pentagon", "hexagon"]
    center_positions, grid_pos = gen_position_description(rules["shapes"])
    special_shapes = {}
    # if(len(rules["shapes"])==1):
    #     grid_pos[0]="center"
    relation_heads = []
    for i in range(len(rules["shapes"])):
        sep_line = grid_pos[i] + " "
        relation_head = grid_pos[i] + " "
        shape = rules["shapes"][i]
        if ("special_info" in shape.keys()) and shape["special_info"] != "":
            sep_line += shape["special_info"][:-2]
            relation_head += shape["special_info"][:-2]
            if shape["type"] == "polygon":
                if "rectangle" in shape["special_info"]:
                    wh = rect_width_height(shape["points"])
                    sep_line += " | width: {:.2f} | height: {:.2f}".format(wh[0], wh[1])
                elif "equilateral triangle" in shape["special_info"]:
                    sep_line += " | side_length: {:.2f}".format(euc_dist(shape["points"][0], shape["points"][1]))
            elif shape["type"] == "ellipse":
                if "circle" in shape["special_info"]:
                    sep_line += " | diameter: {:.2f}".format(shape["major_axis"])
        elif shape["type"] == "polygon":
            if len(shape["points"]) < len(polygon_special):
                sep_line += polygon_special[len(shape["points"])]
                relation_head += polygon_special[len(shape["points"])]
            else:
                sep_line += "polygon | n_sides: {}".format(len(shape["points"]))
                relation_head += "polygon"
        elif shape["type"] == "ellipse":
            sep_line += "ellipse | major_axis: {:.2f} | minor_axis: {:.2f}".format(
                shape["major_axis"], shape["minor_axis"]
            )
            relation_head += "ellipse"
        elif shape["type"] == "segment":
            sep_line += "segment | length: {:.2f}".format(euc_dist(shape["points"][0], shape["points"][1]))
            relation_head += "segment"
        else:
            sep_line += shape["type"]
            relation_head += shape["type"]
        relation_heads.append(relation_head)
        ret_text += sep_line
        special_shape = sep_line.split(" | ")
        if len(special_shape) > 1:
            shape_name = special_shape[0][len(grid_pos[i]) + 1 :]
            if shape_name not in special_shapes.keys():
                special_shapes[shape_name] = {"positions": [], "params": {}}
            special_shapes[shape_name]["positions"].append(center_positions[i])
            for j in range(len(special_shape) - 1):
                param_pair = special_shape[j + 1].split(": ")
                param_name = param_pair[0].replace("_", " ")
                param_value = param_pair[1]
                if param_name not in special_shapes[shape_name]["params"].keys():
                    special_shapes[shape_name]["params"][param_name] = []
                special_shapes[shape_name]["params"][param_name].append(param_value)
        ret_text += "\n"
    ret_text += "END SHAPES\n"
    ret_text += "BEGIN RELATIONS\n"
    for rel in rules["relations"]:
        sep_line = "{}, {} | {}\n".format(relation_heads[rel[0]], relation_heads[rel[1]], rel[2])
        ret_text += sep_line
    ret_text += "END RELATIONS"
    return ret_text, gen_input(special_shapes)


def decide_rel_pos(center_positions, x1, x2, y1, y2):
    rel_positions = []
    for center in center_positions:
        x = center[0]
        y = center[1]
        if x < x1:
            if y < y1:
                rel_positions.append("bottom-left")
            elif y <= y2:
                rel_positions.append("left")
            else:
                rel_positions.append("top-left")
        elif x <= x2:
            if y < y1:
                rel_positions.append("bottom")
            elif y <= y2:
                rel_positions.append("center")
            else:
                rel_positions.append("top")
        else:
            if y < y1:
                rel_positions.append("bottom-right")
            elif y <= y2:
                rel_positions.append("right")
            else:
                rel_positions.append("top-right")
    return rel_positions


def gen_relative_position(center_positions):
    mass_center = points_center(center_positions)
    points_x = [point[0] for point in center_positions]
    points_y = [point[1] for point in center_positions]
    edge_left = min(points_x)
    edge_right = max(points_x)
    edge_top = max(points_y)
    edge_bottom = min(points_y)
    x1 = (edge_left + mass_center[0]) / 2
    x2 = (edge_right + mass_center[0]) / 2
    y1 = (edge_bottom + mass_center[1]) / 2
    y2 = (edge_top + mass_center[1]) / 2
    return decide_rel_pos(center_positions, x1, x2, y1, y2)


def group_by_position(positions):
    count = {}
    for i in range(len(positions)):
        if positions[i] not in count.keys():
            count[positions[i]] = []
        count[positions[i]].append(i)
    return count


def gen_input(special_shapes: dict):
    head = "Please provide a fluent and detailed description of the geometric patterns in this image and their relationships. "
    seed = int(hashlib.md5(json.dumps(special_shapes).encode("utf-8")).hexdigest(), 16)
    random.seed(seed)

    def gen_filter_map():
        filter_map = {}
        for key in special_shapes.keys():
            filter_map[key] = []
            for i in range(len(special_shapes[key]["positions"])):
                rnd = random.random()
                if rnd < caption_args.numeric_ratio:
                    filter_map[key].append(i)
        return filter_map

    def drop_shapes():
        filter_map = gen_filter_map()
        special_shapes_filtered = {}
        is_full = True
        for key in filter_map.keys():
            if len(filter_map[key]) > 0:
                special_shapes_filtered[key] = {"positions": [], "params": {}}
                for arg in special_shapes[key]["params"].keys():
                    special_shapes_filtered[key]["params"][arg] = []
                for idx in filter_map[key]:
                    special_shapes_filtered[key]["positions"].append(special_shapes[key]["positions"][idx])
                    for arg in special_shapes[key]["params"].keys():
                        special_shapes_filtered[key]["params"][arg].append(special_shapes[key]["params"][arg][idx])
                if len(filter_map[key]) < len(special_shapes[key]["positions"]):
                    special_shapes_filtered[key]["is_full"] = False
                    is_full = False
                else:
                    special_shapes_filtered[key]["is_full"] = True
            else:
                is_full = False
        return special_shapes_filtered, is_full, filter_map

    special_shapes_filtered, is_full, filter_map = drop_shapes()

    def choose_head(is_full):
        head_str = ""
        if len(special_shapes.keys()) == 0:
            rnd_start = random.randint(0, len(head_start_no_param_pool) - 1)
            rnd_end = random.randint(0, len(head_end_pool) - 1)
            head_str = head_start_no_param_pool[rnd_start] + head_end_pool[rnd_end]
        else:
            if random.random() > 0.5:
                rnd_start = random.randint(0, len(head_start_no_param_pool) - 1)
                rnd_part1 = random.randint(0, len(head_with_param_part1_pool) - 1)
                rnd_end = random.randint(0, len(head_end_pool) - 1)
                head_str = (
                    head_start_no_param_pool[rnd_start] + head_with_param_part1_pool[rnd_part1] + head_end_pool[rnd_end]
                )
            else:
                rnd_start_part1 = random.randint(0, len(head_start_with_param_part1_pool) - 1)
                rnd_start_part2 = random.randint(0, len(head_start_with_param_part2_pool) - 1)
                rnd_end = random.randint(0, len(head_end_pool) - 1)
                head_str = (
                    head_start_with_param_part1_pool[rnd_start_part1]
                    + head_start_with_param_part2_pool[rnd_start_part2]
                    + head_end_pool[rnd_end]
                )
        if (not is_full) and (len(special_shapes_filtered.keys()) != 0):
            rnd_not_full_part1 = random.randint(0, len(head_not_full_part1_pool) - 1)
            rnd_not_full_part2 = random.randint(0, len(head_not_full_part2_pool) - 1)
            head_str += head_not_full_part1_pool[rnd_not_full_part1] + head_not_full_part2_pool[rnd_not_full_part2]
        return head_str

    head = choose_head(is_full)
    single_1_param_format = "The {attr} of the {shape} is {value}. "
    single_2_param_format = "The {attr1} and {attr2} of the {shape} are {value1} and {value2}, respectively. "
    single_concentric_1_param_format = "The {attr}s of the {shape}s are {value}, respectively. "
    single_concentric_2_param_format = "The {attr1}s of the {shape}s are {value1}, respectively, and their {attr2}s correspond to {value2}, respectively. "

    plural_format_head = "For the {shape}s, "
    plural_1_param_format_part = "the {attr} of the {pos} {shape} is {value}, "
    plural_1_param_format_end = "and the {attr} of the {pos} {shape} is {value}. "
    plural_concentric_1_param_format_part = "the {attr}s of the {pos} {shape}s are {value}, "
    plural_concentric_1_param_format_end = "and the {attr}s of the {pos} {shape}s are {value}, respectively. "

    plural_2_param_format_part = "the {attr1} and {attr2} of the {pos} {shape} are {value1} and {value2}, "
    plural_2_param_format_end = (
        "and the {attr1} and {attr2} of the {pos} {shape} are {value1} and {value2}, respectively. "
    )
    plural_concentric_2_param_format_part = (
        "the {attr1}s of the {pos} {shape}s are {value1}, while their {attr2}s correspond to {value2}, respectively. "
    )
    plural_concentric_2_param_format_end = "and the {attr1}s of the {pos} {shape}s are {value1}, while their {attr2}s correspond to {value2}, respectively. "

    def expand_values(values_list):
        expanded_str = ""
        for idx in range(len(values_list)):
            if idx == 0:
                expanded_str += str(values_list[idx])
            elif idx == len(values_list) - 1:
                expanded_str += " and " + str(values_list[idx])
            else:
                expanded_str += ", " + str(values_list[idx])
        return expanded_str

    final_str = ""
    final_str += head
    for key in special_shapes_filtered.keys():  # 特殊形状名称
        rel_pos = gen_relative_position(special_shapes[key]["positions"])
        rel_pos = [rel_pos[idx] for idx in filter_map[key]]
        grouped_pos = group_by_position(rel_pos)
        if (len(grouped_pos.keys()) == 1) and special_shapes_filtered[key]["is_full"]:  # 只有一种位置关系时不用指明位置
            for pos_name in grouped_pos.keys():
                if len(grouped_pos[pos_name]) == 1:  # 只有一个形状
                    param_names = list(special_shapes_filtered[key]["params"].keys())
                    if len(param_names) == 1:
                        value = special_shapes_filtered[key]["params"][param_names[0]][grouped_pos[pos_name][0]]
                        final_str += single_1_param_format.format(attr=param_names[0], shape=key, value=value)
                    elif len(param_names) == 2:
                        value1 = special_shapes_filtered[key]["params"][param_names[0]][grouped_pos[pos_name][0]]
                        value2 = special_shapes_filtered[key]["params"][param_names[1]][grouped_pos[pos_name][0]]
                        final_str += single_2_param_format.format(
                            attr1=param_names[0], attr2=param_names[1], shape=key, value1=value1, value2=value2
                        )
                else:  # 同心
                    param_names = list(special_shapes_filtered[key]["params"].keys())
                    if len(param_names) == 1:
                        value = [
                            special_shapes_filtered[key]["params"][param_names[0]][idx] for idx in grouped_pos[pos_name]
                        ]
                        final_str += single_concentric_1_param_format.format(
                            attr=param_names[0], shape=key, value=expand_values(value)
                        )
                    elif len(param_names) == 2:
                        value1 = [
                            special_shapes_filtered[key]["params"][param_names[0]][idx] for idx in grouped_pos[pos_name]
                        ]
                        value2 = [
                            special_shapes_filtered[key]["params"][param_names[1]][idx] for idx in grouped_pos[pos_name]
                        ]
                        final_str += single_concentric_2_param_format.format(
                            attr1=param_names[0],
                            attr2=param_names[1],
                            shape=key,
                            value1=expand_values(value1),
                            value2=expand_values(value2),
                        )
        else:
            final_str += plural_format_head.format(shape=key)
            for pos_idx in range(len(list(grouped_pos.keys()))):
                pos_name = list(grouped_pos.keys())[pos_idx]
                if len(grouped_pos[pos_name]) == 1:
                    param_names = list(special_shapes_filtered[key]["params"].keys())
                    if len(param_names) == 1:
                        value = special_shapes_filtered[key]["params"][param_names[0]][grouped_pos[pos_name][0]]
                        format_str_switch = plural_1_param_format_part
                        if pos_idx == len(list(grouped_pos.keys())) - 1:
                            format_str_switch = plural_1_param_format_end
                        final_str += format_str_switch.format(attr=param_names[0], pos=pos_name, shape=key, value=value)
                    elif len(param_names) == 2:
                        value1 = special_shapes_filtered[key]["params"][param_names[0]][grouped_pos[pos_name][0]]
                        value2 = special_shapes_filtered[key]["params"][param_names[1]][grouped_pos[pos_name][0]]
                        format_str_switch = plural_2_param_format_part
                        if (pos_idx == len(list(grouped_pos.keys())) - 1) and (len(list(grouped_pos.keys())) > 1):
                            format_str_switch = plural_2_param_format_end
                        elif (pos_idx == len(list(grouped_pos.keys())) - 1) and (len(list(grouped_pos.keys())) == 1):
                            format_str_switch = plural_2_param_format_part[:-2] + ". "
                        final_str += format_str_switch.format(
                            attr1=param_names[0],
                            attr2=param_names[1],
                            pos=pos_name,
                            shape=key,
                            value1=value1,
                            value2=value2,
                        )
                else:
                    param_names = list(special_shapes_filtered[key]["params"].keys())
                    if len(param_names) == 1:
                        value = [
                            special_shapes_filtered[key]["params"][param_names[0]][idx] for idx in grouped_pos[pos_name]
                        ]
                        format_str_switch = plural_concentric_1_param_format_part
                        if pos_idx == len(list(grouped_pos.keys())) - 1:
                            format_str_switch = plural_concentric_1_param_format_end
                        final_str += format_str_switch.format(
                            attr=param_names[0], pos=pos_name, shape=key, value=expand_values(value)
                        )
                    elif len(param_names) == 2:
                        value1 = [
                            special_shapes_filtered[key]["params"][param_names[0]][idx] for idx in grouped_pos[pos_name]
                        ]
                        value2 = [
                            special_shapes_filtered[key]["params"][param_names[1]][idx] for idx in grouped_pos[pos_name]
                        ]
                        format_str_switch = plural_concentric_2_param_format_part
                        if pos_idx == len(list(grouped_pos.keys())) - 1:
                            format_str_switch = plural_concentric_2_param_format_end
                        final_str += format_str_switch.format(
                            attr1=param_names[0],
                            attr2=param_names[1],
                            pos=pos_name,
                            shape=key,
                            value1=expand_values(value1),
                            value2=expand_values(value2),
                        )
    return final_str


def main():
    model_name, model_size = caption_args.caption_llm.split("-")
    generator = generator_mapping[model_name](model_path_mapping[model_name].format(size=model_size))
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)[run_args.start_pos : run_args.end_pos]
    caption(samples, generator, data_args.caption_path)


if __name__ == "__main__":
    main()

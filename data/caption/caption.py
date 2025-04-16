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
from data.caption.prompt import *

center_size_ratio = 1 / 3


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
        rule_skeleton, data_input = gen_user_input_skeleton(rule, caption_args.numeric_ratio)
        rule_str = json.dumps(rule_skeleton)
        rule_strs.append(rule_str)
        input_texts.append(data_input)

    if caption_args.debug_option == "":
        messages = [
            [
                {"role": "system", "content": context},
                {"role": "user", "content": rule_strs[i]},
                {"role": "user", "content": "The descriptive text you should generate:"},
            ]
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
    elif "nollm" in caption_args.debug_option:
        with open(output_path, "w") as f:
            for batch_idx, item in tqdm(enumerate(zip(input_texts, rule_strs))):
                f.write(json.dumps({"input": item[0], "output": item[1]}) + "\n")
                f.flush()

    # with open(output_path, "w") as f:
    #     for input in rule_strs:
    #         f.write(json.dumps({"input": input}) + "\n")
    #         f.flush()


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


def get_center_and_weight(shapez, n_digits=4):
    shape = shapez
    if "concentric" in shapez.keys():
        shape = list(shapez["concentric"].values())[-1]
    if shape["type"] in ["segment", "polygon"]:
        p_center = points_center(shape["points"])
        return [
            round(p_center[0], ndigits=n_digits),
            round(p_center[1], ndigits=n_digits),
        ], calculate_polygon_C(shape["points"])
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
                abs_positions.append("bottom-left")
            elif y <= y2:
                abs_positions.append("left")
            else:
                abs_positions.append("top-left")
        elif x <= x2:
            if y < y1:
                abs_positions.append("bottom")
            elif y <= y2:
                abs_positions.append("center")
            else:
                abs_positions.append("top")
        else:
            if y < y1:
                abs_positions.append("bottom-right")
            elif y <= y2:
                abs_positions.append("right")
            else:
                abs_positions.append("top-right")
    return abs_positions


def gen_grid(shapes, x1, x2, y1, y2):
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
    # points_x = [point[0] for point in center_positions]
    # points_y = [point[1] for point in center_positions]
    # edge_left = min(points_x)
    # edge_right = max(points_x)
    # edge_top = max(points_y)
    # edge_bottom = min(points_y)
    # x1 = (edge_left + mass_center[0]) / 2
    # x2 = (edge_right + mass_center[0]) / 2
    # y1 = (edge_bottom + mass_center[1]) / 2
    # y2 = (edge_top + mass_center[1]) / 2
    # [0,1] range canvas
    # x1=1/3
    # x2=2/3
    # y1=1/3
    # y2=2/3
    return center_positions, weights, decide_abs_pos(center_positions, x1, x2, y1, y2)


def gen_position_description(shapes):
    return gen_grid(shapes, 1 / 3, 2 / 3, 1 / 3, 2 / 3)


def refine_relation(shapes, relation):
    if "concentric" in relation[-1]:
        return None, "", None
    trans_table = {
        "tangent": "is tangent to",
        "parallel": "is parallel to",
        "similar": "is similar to",
        "symmetric": "is symmetric with",
        "circumscribed": "is circumscribed around",
        "inscribed": "is inscribed within",
        "shared edge": "has a shared edge with",
        "diagonal": "is the diagonal of",
        "major axis": "is the major axis of",
        "minor axis": "is the minor axis of",
        "diameter": "is the diameter of",
        "internal tangent": "is inscribed within",
        "external tangent": "is circumscribed around",
    }
    operator = ""
    for k in trans_table.keys():
        if k in relation[-1]:
            operator = trans_table[k]
    dst = relation[1]
    src = relation[0]
    if (
        ("inscribed circle" in relation[-1])
        or ("circumscribed circle" in relation[-1])
        or ("tangent circle" in relation[-1])
    ):
        if "circle" in shapes[src]["name"]:
            pass
        else:
            src, dst = dst, src
    elif "tangent line" in relation[-1]:
        if (
            ("line" in shapes[src]["name"])
            or ("ray" in shapes[src]["name"])
            or ("segment" in shapes[src]["name"])
        ):
            pass
        else:
            src, dst = dst, src
    elif ("major axis" in relation[-1]) or ("minor axis" in relation[-1]) or ("diameter" in relation[-1]):
        if "segment" in shapes[src]["name"]:
            pass
        else:
            src, dst = dst, src
    return src, operator, dst


def patch_relations(rules, skeleton):
    def get_path_of_shape(temp, id, path: list, success: list):
        if success[0]:
            return
        temp_type = type(temp)
        if temp_type is list:
            for sub in temp:
                get_path_of_shape(sub, id, path, success)
                if success[0]:
                    return
        elif temp_type is dict:
            if "concentric" in temp:
                path.append("concentric")
                get_path_of_shape(temp["concentric"], id, path, success)
                if success[0]:
                    return
                path.pop()
            elif "id" in temp:
                if temp["id"] == id:
                    path.append(temp["name"])
                    success[0] = True
                    return
            else:
                for k in temp.keys():
                    path.append(k)
                    get_path_of_shape(temp[k], id, path, success)
                    if success[0]:
                        return
                    path.pop()

    def gen_selector_str(path):
        s = ""
        for p in range(len(path)):
            if p != 0:
                s += "::" + path[p]
            else:
                s += path[p]
        return s

    for relation in rules["relations"]:
        if "relations" not in skeleton:
            skeleton["relations"] = []
        relation_str = ""
        src, mid, dst = refine_relation(rules["shapes"], relation)
        if src is None:
            continue
        path = ["canvas"]
        success = [False]
        get_path_of_shape(skeleton["canvas"], src, path, success)
        relation_str += gen_selector_str(path)
        relation_str += " " + mid + " "
        path = ["canvas"]
        success = [False]
        get_path_of_shape(skeleton["canvas"], dst, path, success)
        relation_str += gen_selector_str(path)
        skeleton["relations"].append(relation_str)


def choose_grid(pos, x1, x2, y1, y2, xbase, ybase):
    new_xbase = 0
    new_ybase = 0
    x_tr = 0
    y_tr = 0
    if pos == "top-left":
        new_xbase = xbase
        new_ybase = y2
        x_tr = x1
        y_tr = y2 + y1 - ybase
    elif pos == "top":
        new_xbase = x1
        new_ybase = y2
        x_tr = x2
        y_tr = y2 + y1 - ybase
    elif pos == "top-right":
        new_xbase = x2
        new_ybase = y2
        x_tr = x2 + x1 - xbase
        y_tr = y2 + y1 - ybase
    elif pos == "left":
        new_xbase = xbase
        new_ybase = y1
        x_tr = x1
        y_tr = y2
    elif pos == "center":
        new_xbase = x1
        new_ybase = y1
        x_tr = x2
        y_tr = y2
    elif pos == "right":
        new_xbase = x2
        new_ybase = y1
        x_tr = x2 + x1 - xbase
        y_tr = y2
    elif pos == "bottom-left":
        new_xbase = xbase
        new_ybase = ybase
        x_tr = x1
        y_tr = y1
    elif pos == "bottom":
        new_xbase = x1
        new_ybase = ybase
        x_tr = x2
        y_tr = y1
    elif pos == "bottom-right":
        new_xbase = x2
        new_ybase = ybase
        x_tr = x2 + x1 - xbase
        y_tr = y1
    x_center_l = (x_tr - new_xbase) * center_size_ratio
    y_center_l = (y_tr - new_ybase) * center_size_ratio
    return (
        new_xbase + (x_tr - new_xbase - x_center_l) / 2,
        x_tr - (x_tr - new_xbase - x_center_l) / 2,
        new_ybase + (y_tr - new_ybase - y_center_l) / 2,
        y_tr - (y_tr - new_ybase - y_center_l) / 2,
        new_xbase,
        new_ybase,
    )


def sort_shapes_by_weight(shapes):
    return sorted(shapes, key=lambda x: x["weight"])


def disambiguation(canvas_part, x1, x2, y1, y2, xbase, ybase, depth):
    # 处理同心组
    concentric_groups_raw = {}
    for shape in canvas_part:
        cp = str(shape["center_position"])
        if cp not in concentric_groups_raw:
            concentric_groups_raw[cp] = []
        concentric_groups_raw[cp].append(shape)
    concentric_groups = {}
    for k in concentric_groups_raw.keys():
        if len(concentric_groups_raw[k]) > 1:
            concentric_groups[k] = concentric_groups_raw[k]
    # 处理同名形状
    same_shape_groups_raw = {}
    for shape in canvas_part:
        if "concentric" in shape:
            temp_shape_names = []
            for v in shape["concentric"].values():
                if v["name"] not in temp_shape_names:
                    temp_shape_names.append(v["name"])
                    if v["name"] not in same_shape_groups_raw:
                        same_shape_groups_raw[v["name"]] = []
                    same_shape_groups_raw[v["name"]].append(shape)
        else:
            if shape["name"] not in same_shape_groups_raw:
                same_shape_groups_raw[shape["name"]] = []
            same_shape_groups_raw[shape["name"]].append(shape)
    same_shape_groups = {}
    for k in same_shape_groups_raw.keys():
        if len(same_shape_groups_raw[k]) > 1:
            same_shape_groups[k] = same_shape_groups_raw[k]
    canvas_part_remain = []
    for shape in canvas_part:
        is_in_concentric_group = False
        for k in concentric_groups.keys():
            for shape2 in concentric_groups[k]:
                if shape["id"] == shape2["id"]:
                    is_in_concentric_group = True
                    break
            if is_in_concentric_group:
                break
        is_in_same_shape_group = False
        # for k in same_shape_groups.keys():
        #     for shape2 in same_shape_groups[k]:
        #         if shape["id"]==shape2["id"]:
        #             is_in_same_shape_group=True
        #             break
        #     if is_in_same_shape_group:
        #         break
        if (not is_in_concentric_group) and (not is_in_same_shape_group):
            canvas_part_remain.append(shape)
    if len(concentric_groups) > 0:  # 有同心结构
        center_positions, weights, grid_posz = gen_grid(canvas_part, x1, x2, y1, y2)
        grid_pos = {}
        for i in range(len(canvas_part)):
            grid_pos[canvas_part[i]["id"]] = grid_posz[i]
        sub_part = {}
        for k in concentric_groups.keys():
            concentric_part = {
                "concentric": {},
                "center_position": concentric_groups[k][0]["center_position"],
                "id": "",
            }
            pos = grid_pos[concentric_groups[k][0]["id"]]
            if pos not in sub_part:
                sub_part[pos] = []
            concentric_groups[k] = sort_shapes_by_weight(concentric_groups[k])
            for j in range(len(concentric_groups[k])):
                concentric_part["concentric"][
                    "the " + ordinal_numbers[j] + " shape from the inside, counting outward"
                ] = concentric_groups[k][j]
                concentric_part["id"] += str(concentric_groups[k][j]["id"]) + "/"
            sub_part[pos].append(concentric_part)
        for shape_remain in canvas_part_remain:
            pos = grid_pos[shape_remain["id"]]
            if pos not in sub_part:
                sub_part[pos] = []
            sub_part[pos].append(shape_remain)
        for k in sub_part.keys():
            new_x1, new_x2, new_y1, new_y2, new_xbase, new_ybase = choose_grid(
                k, x1, x2, y1, y2, xbase, ybase
            )
            sub_part[k] = disambiguation(
                sub_part[k], new_x1, new_x2, new_y1, new_y2, new_xbase, new_ybase, depth + 1
            )
        return sub_part
    elif len(same_shape_groups) > 0:  # 有同名形状
        center_positions, weights, grid_posz = gen_grid(canvas_part, x1, x2, y1, y2)
        grid_pos = {}
        for i in range(len(canvas_part)):
            grid_pos[canvas_part[i]["id"]] = grid_posz[i]
        sub_part = {}
        for j in range(len(canvas_part)):
            pos = grid_posz[j]
            if pos not in sub_part:
                sub_part[pos] = []
            sub_part[pos].append(canvas_part[j])
        for k in sub_part.keys():
            new_x1, new_x2, new_y1, new_y2, new_xbase, new_ybase = choose_grid(
                k, x1, x2, y1, y2, xbase, ybase
            )
            sub_part[k] = disambiguation(
                sub_part[k], new_x1, new_x2, new_y1, new_y2, new_xbase, new_ybase, depth + 1
            )
        return sub_part
    elif depth == 0:
        center_positions, weights, grid_posz = gen_grid(canvas_part, x1, x2, y1, y2)
        grid_pos = {}
        for i in range(len(canvas_part)):
            grid_pos[canvas_part[i]["id"]] = grid_posz[i]
        sub_part = {}
        for j in range(len(canvas_part)):
            pos = grid_posz[j]
            if pos not in sub_part:
                sub_part[pos] = []
            sub_part[pos].append(canvas_part[j])
        return sub_part
    else:
        return canvas_part


def refine_special_info(special_info):
    args = special_info.split(". ")
    return args[0]


def gen_user_input_skeleton(rules, numeric_ratio):
    skeleton = {"total": 0, "canvas": {}}
    polygon_special = [0, 0, 0, "triangle", "quadrilateral", "pentagon", "hexagon"]
    x1 = (1 - center_size_ratio) / 2
    x2 = 1 - (1 - center_size_ratio) / 2
    y1 = x1
    y2 = x2
    center_positions, weights, grid_pos = gen_grid(rules["shapes"], x1, x2, y1, y2)
    skeleton["total"] = len(rules["shapes"])
    for i in range(len(rules["shapes"])):
        rules["shapes"][i]["weight"] = weights[i]
        rules["shapes"][i]["id"] = i
        rules["shapes"][i]["center_position"] = center_positions[i]
        rules["shapes"][i]["name"] = rules["shapes"][i]["type"]
        if rules["shapes"][i]["type"] == "polygon":
            if len(rules["shapes"][i]["points"]) < len(polygon_special):
                rules["shapes"][i]["name"] = polygon_special[len(rules["shapes"][i]["points"])]
        if "special_info" in rules["shapes"][i]:
            if rules["shapes"][i]["special_info"] != "":
                rules["shapes"][i]["name"] = refine_special_info(rules["shapes"][i]["special_info"])
    skeleton["canvas"] = disambiguation(rules["shapes"], x1, x2, y1, y2, 0, 0, 0)
    patch_relations(rules, skeleton)
    simplify_skeleton(rules, skeleton, numeric_ratio)
    data_input_str = gen_data_input(skeleton)
    return skeleton, data_input_str


def cut2f(num):
    return float("{:.2f}".format(num))


def get_numeric_info(shape):
    name = shape["name"]
    if "rectangle" in name:
        wh = rect_width_height(shape["points"])
        return {"width": cut2f(wh[0]), "height": cut2f(wh[1])}
    elif "square" in name:
        wh = rect_width_height(shape["points"])
        return {"side length": cut2f(wh[0])}
    elif "equilateral triangle" in name:
        return {"side length": cut2f(euc_dist(shape["points"][0], shape["points"][1]))}
    elif "circle" in name:
        return {"diameter": cut2f(shape["major_axis"])}
    elif "polygon" in name:
        return {"number of sides": len(shape["points"])}
    elif "ellipse" in name:
        return {"major axis": cut2f(shape["major_axis"]), "minor axis": cut2f(shape["minor_axis"])}
    elif "segment" in name:
        return {"length": cut2f(euc_dist(shape["points"][0], shape["points"][1]))}
    return {}


def simplify_skeleton(rules, skeleton, numeric_ratio):
    seed = int(hashlib.md5(json.dumps(rules).encode("utf-8")).hexdigest(), 16)
    random.seed(seed)

    def simplify_skeleton_recursive_part(parent, idx, temp):
        temp_type = type(temp)
        if temp_type is list:
            for i in range(len(temp)):
                simplify_skeleton_recursive_part(temp, i, temp[i])
        elif temp_type is dict:
            if "concentric" in temp:
                temp = {"concentric": temp["concentric"]}
                parent[idx] = temp
                simplify_skeleton_recursive_part(temp, "concentric", temp["concentric"])
            elif "name" in temp:
                simplified = {"name": temp["name"]}
                if "relations" in temp:
                    simplified["relations"] = temp["relations"]
                rnd = random.random()
                if rnd < numeric_ratio:
                    numeric_info = get_numeric_info(temp)
                    simplified.update(numeric_info)
                parent[idx] = simplified
            else:
                for k in temp.keys():
                    simplify_skeleton_recursive_part(temp, k, temp[k])

    simplify_skeleton_recursive_part(None, None, skeleton)


def decide_rel_pos(center_positions):
    points_x = [point[0] for point in center_positions]
    points_y = [point[1] for point in center_positions]
    edge_left = min(points_x)
    edge_right = max(points_x)
    edge_top = max(points_y)
    edge_bottom = min(points_y)
    x_center_l = (edge_right - edge_left) * center_size_ratio
    y_center_l = (edge_top - edge_bottom) * center_size_ratio
    x1 = edge_left + (edge_right - edge_left - x_center_l) / 2
    x2 = edge_right - (edge_right - edge_left - x_center_l) / 2
    y1 = edge_bottom + (edge_top - edge_bottom - y_center_l) / 2
    y2 = edge_top - (edge_top - edge_bottom - y_center_l) / 2
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
    # mass_center = points_center(center_positions)
    # points_x = [point[0] for point in center_positions]
    # points_y = [point[1] for point in center_positions]
    # edge_left = min(points_x)
    # edge_right = max(points_x)
    # edge_top = max(points_y)
    # edge_bottom = min(points_y)
    # x1 = (edge_left + mass_center[0]) / 2
    # x2 = (edge_right + mass_center[0]) / 2
    # y1 = (edge_bottom + mass_center[1]) / 2
    # y2 = (edge_top + mass_center[1]) / 2
    return decide_rel_pos(center_positions)


def group_by_position(positions):
    count = {}
    for i in range(len(positions)):
        if positions[i] not in count.keys():
            count[positions[i]] = []
        count[positions[i]].append(i)
    return count


def gen_data_input(skeleton):
    single_1_param_format = "The {attr} of {shape} is {value}. "
    single_2_param_format = "The {attr1} and {attr2} of {shape} are {value1} and {value2}, respectively. "

    def gen_path_recursive(temp, path: list, results):
        temp_type = type(temp)
        if temp_type is list:
            for sub in temp:
                gen_path_recursive(sub, path, results)
        elif temp_type is dict:
            if "concentric" in temp:
                path.append("concentric")
                gen_path_recursive(temp["concentric"], path, results)
                path.pop()
            elif "name" in temp:
                param_n = []
                param_v = []
                for k in temp.keys():
                    if k != "name":
                        param_n.append(k)
                        param_v.append(temp[k])
                path_str = '"'
                for p in path:
                    path_str += p + "::"
                path_str += temp["name"]
                path_str += '"'
                if len(param_n) == 1:
                    info_str = single_1_param_format.format(attr=param_n[0], shape=path_str, value=param_v[0])
                    results.append(info_str)
                elif len(param_n) == 2:
                    info_str = single_2_param_format.format(
                        attr1=param_n[0],
                        attr2=param_n[1],
                        shape=path_str,
                        value1=param_v[0],
                        value2=param_v[1],
                    )
                    results.append(info_str)
            else:
                for k in temp.keys():
                    path.append(k)
                    gen_path_recursive(temp[k], path, results)
                    path.pop()

    path = ["canvas"]
    results = []
    gen_path_recursive(skeleton["canvas"], path, results)
    result_str = ""
    for s in results:
        result_str += s

    def choose_head(results):
        head_str = ""
        if len(results) == 0:
            rnd_start = random.randint(0, len(head_start_no_param_pool) - 1)
            rnd_end = random.randint(0, len(head_end_pool) - 1)
            head_str = head_start_no_param_pool[rnd_start] + head_end_pool[rnd_end]
        else:
            if random.random() > 0.5:
                rnd_start = random.randint(0, len(head_start_no_param_pool) - 1)
                rnd_part1 = random.randint(0, len(head_with_param_part1_pool) - 1)
                rnd_end = random.randint(0, len(head_end_pool) - 1)
                head_str = (
                    head_start_no_param_pool[rnd_start]
                    + head_with_param_part1_pool[rnd_part1]
                    + head_end_pool[rnd_end]
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
        return head_str

    result_str = (choose_head(results) + result_str).strip()
    return result_str


def main():
    if data_args.stage == 1:
        if caption_args.debug_option == "":
            model_name, model_id = caption_args.caption_llm.split("-", 1)
            generator = generator_mapping[model_name](model_path_mapping[model_name].format(model_id))
            # generator=None
            with open(data_args.rules_path, "r") as f:
                samples = json.load(f)[run_args.start_pos : run_args.end_pos]
            os.makedirs(data_args.caption_dir, exist_ok=True)
            caption(samples, generator, data_args.caption_path)  # type: ignore
        elif "nollm" in caption_args.debug_option:
            with open(data_args.rules_path, "r") as f:
                samples = json.load(f)[run_args.start_pos : run_args.end_pos]
            os.makedirs(data_args.caption_dir, exist_ok=True)
            caption(samples, None, data_args.caption_path)  # type: ignore
    elif data_args.stage == 2:
        import data.caption.caption2

        data.caption.caption2.main()


if __name__ == "__main__":
    main()

"construct descriptions according to generated rules"

import hashlib
import json
import random
import re
from typing import Any

from tqdm import tqdm

from common.args import data_args, run_args
from data.caption.caption_2nd.chomata import Chomata
from data.caption.caption_2nd.deposit import Deposit
from data.caption.caption_2nd.proloculus import Proloculus
from data.caption.caption_2nd.septa import Septa
from data.caption.caption_2nd.shell import Shell
from data.caption.caption_2nd.tunnel import Tunnel
from data.caption.caption_2nd.volution import Volution
from data.caption.prompt import *

import matplotlib.pyplot as plt
import numpy as np


def caption(rules: list[dict[str, Any]], generator, output_path: str):
    # TODO generate captions. Use whatever technique you like, e.g., demonstrations, cot, ...
    # TODO also consider how to handle distance:
    #      It should be mentioned in input prompt. Try scaling to produce different distances.
    # NOTE see common/llm.py for the use of open-source LLMs.
    # test='''[{"type": "line", "points": [[0, 0], [3, 3]]},{"type": "polygon", "points": [[1, 1], [2, 1], [1, 2], [2, 2]]},{"type": "polygon", "points": [[0, 10], [-5, 5], [5, 5]]}]'''
    # rule_str=json.dumps(rules)
    # rules=rules[100:200]
    rule_strs = []
    # input_texts: list[str] = []
    idx = 0
    for rule in tqdm(rules):
        input_str, rule_str = gen_user_input_txt_2nd(rule)
        rule_strs.append({"input": input_str, "output": rule_str})
        idx += 1

    # caption2_debug()

    with open(output_path, "w") as f:
        f.write(json.dumps(rule_strs, indent=4))
        f.flush()

septa_diffs=[]

def plot_histogram(arr, bins='auto', save_path='dataset/histogram.png', 
                   title='Frequency Distribution', xlabel='Value', ylabel='Frequency'):
    """
    绘制数组的频率分布直方图并保存
    
    参数:
    arr -- 输入数组
    bins -- 直方图的柱子数量，可以是整数或'auto'、'sturges'等自动计算算法 (默认 'auto')
    save_path -- 保存的文件路径 (默认 'histogram.png')
    title -- 图表标题 (默认 'Frequency Distribution')
    xlabel -- X轴标签 (默认 'Value')
    ylabel -- Y轴标签 (默认 'Frequency')
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图
    plt.hist(arr, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 添加基本元素
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # 添加网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加均值和中位数线
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1.5, 
                label=f'Mean: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle='dashed', linewidth=1.5, 
                label=f'Median: {median_val:.2f}')
    
    # 添加图例
    plt.legend()
    
    # 自动调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭图形以释放内存
    
    print(f"直方图已保存至: {save_path}")

def caption2_debug():
    plot_histogram(septa_diffs)

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

    random.seed(hashlib.md5(json.dumps(rule).encode()).hexdigest())

    random_pixel_div_mm_offset = random.randint(-40, 150)

    obj_parts.append(
        Shell(
            volution_max["type"],
            volution_max["ratio"],
            volution_max["width"],
            volution_max["height"],
            volution_max["control_points"],
            volution_max["vertices"],
            initial_chamber["center"],
            random_pixel_div_mm_offset=random_pixel_div_mm_offset,
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
            random_pixel_div_mm_offset=random_pixel_div_mm_offset,
        )
    )

    obj_parts.append(
        Proloculus(
            "",
            initial_chamber,
            (initial_chamber["major_axis"] + initial_chamber["minor_axis"]) / 2,
            random_pixel_div_mm_offset=random_pixel_div_mm_offset,
        )
    )
    # if len(chomata_shapes) > 0:
    obj_parts.append(Chomata(chomata_shapes, rule["numerical_info"]["num_volutions"], volutions))
    volution_sizes = []
    for i in range(len(obj_parts[-1].volution_heights)):
        volution_sizes.append(obj_parts[-1].volution_heights[i]*obj_parts[-1].volution_widths[i])
    if "tunnel_angles" in rule["numerical_info"] and len(rule["numerical_info"]["tunnel_angles"]) > 0:
        obj_parts.append(
            Tunnel(
                rule,
                rule["numerical_info"]["visible_chomata_idx"],
                obj_parts[-1].chomata_whs_relative,
                obj_parts[-1].chomata_pos_ordered,
                rule["numerical_info"]["tunnel_angles"],
            )
        )
    if "axial_filling" in rule and len(rule["axial_filling"]) > 0:
        obj_parts.append(Deposit(rule["axial_filling"], rule["numerical_info"]["num_volutions"]))
    else:
        obj_parts.append(Deposit([], rule["numerical_info"]["num_volutions"]))
    obj_parts.append(Septa(rule["septa_folds"], volution_sizes))
    
    txt2 = head_start_2nd + "\n"
    feature_tagged = []
    for part in obj_parts:
        feature_tagged.extend(part.genUserInput())
        txt2 += part.genInput() + ""
    feature_tagged.sort(key=featureSortFunc)
    for feat in feature_tagged:
        txt += feat

    # septa_diffs.append(obj_parts[-1].size_diff)

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

import argparse
import json
import os
import random
import re
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from diffuser import diffuse
from figure import Figure_Engine
from PIL import Image
from tqdm import tqdm


def generate_basic_shape(shapes: list, ni: dict) -> tuple:
    figure = Figure_Engine(max_volution=int(ni["num_volutions"]), center=ni["center"], xkcd=True)
    fixed_color = np.abs(np.random.randn(3) / 10 + np.array((0.1, 0.1, 0.1)))
    face_color = np.array([fixed_color[0], fixed_color[1], fixed_color[2], 1])
    for shape in shapes:
        figure.draw(shape, color=fixed_color, trans=face_color)
    basic_img = figure.transfer_to_cv2()
    figure.close()
    return basic_img, figure.volution_memory, figure.max_volution


def redraw_basic_shapes(dif_pic: np.ndarray, shapes: list) -> np.ndarray:
    figure = Figure_Engine(xkcd=True)
    fixed_color = np.abs(np.random.randn(3) / 10 + np.array((0.1, 0.1, 0.1)))
    face_color = np.array([fixed_color[0], fixed_color[1], fixed_color[2], 1])
    for shape in shapes:
        figure.draw(shape, color=fixed_color, trans=face_color)
    redrawn = figure.transfer_to_cv2()
    redrawn = redrawn[:, :, 0:3]
    figure.close()
    return np.minimum(redrawn, dif_pic)


def generate_basic_mask(volution_memory: dict, filling: list, mode = None, debug=None) -> np.ndarray:
    from bisect import bisect

    mask = plt.figure(figsize=(12.8, 12.8), dpi=100)
    ax = mask.add_subplot()
    plt.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # volution_memory structure:
    # -1 -> initial chamber, (x,y) as the middle point
    # other numbers -> volutions, (x, y, angle) as the points of the volutions, sorted by angle[0:2pi~6.28]
    for fill in filling:  # experiment suggests that diffusing these together works better
        start_angle = fill["start_angle"]
        if start_angle < 0:
            start_angle += 2 * np.pi
        end_angle = fill["end_angle"]
        if end_angle < 0:
            end_angle += 2 * np.pi
        start_volution = volution_memory[fill["start_volution"]]  # shape: x,y,a
        end_volution = volution_memory[fill["end_volution"]]
        if start_angle < end_angle:
            angles = np.linspace(start_angle, end_angle, 1200)
        else:  # start_angle > end_angle, ignore ==
            angles1 = np.linspace(start_angle, np.pi * 2, 600)
            angles2 = np.linspace(0, end_angle, 600)
            angles = np.concatenate([angles1, angles2])
        for angle in angles:
            if angle < 0:
                angle += 2 * np.pi
            elif angle > 2 * np.pi:
                angle -= 2 * np.pi
            index_at_start = bisect(start_volution[2], angle)
            index_at_end = bisect(end_volution[2], angle)
            try:
                ax.plot(
                    [start_volution[0][index_at_start], end_volution[0][index_at_end]],  # x
                    [start_volution[1][index_at_start], end_volution[1][index_at_end]],  # y
                    color="black",
                    linewidth=5,
                )
            except IndexError:
                ax.plot(
                    [start_volution[0][index_at_start - 1], end_volution[0][index_at_end - 1]],  # x
                    [start_volution[1][index_at_start - 1], end_volution[1][index_at_end - 1]],  # y
                    color="black",
                    linewidth=5,
                )
    if debug is not None:
        mask.savefig(f"{debug}_Mask_{mode}.png")
    buf = BytesIO()
    mask.savefig(buf, format="png")
    buf.seek(0)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    plt.close(mask)

    return img


def generate_basic_shape_separately(shapes: list, ni: dict):
    figure = Figure_Engine(max_volution=int(ni["num_volutions"]), center=ni["center"])
    fixed_color = np.abs(np.random.randn(3) / 10 + np.array((0.1, 0.1, 0.1)))
    face_color = np.array([fixed_color[0], fixed_color[1], fixed_color[2], 1])
    volution_max = {}
    filtered_shapes = []

    def get_volution_index(shape):
        a = shape["special_info"]
        return int(a[len("volution ") :])

    for shape in shapes:
        if re.match("volution [0-9]+", shape["special_info"]) is not None:
            if volution_max == {}:
                volution_max = shape
            else:
                if get_volution_index(shape) > get_volution_index(volution_max):
                    volution_max = shape
        elif re.match("initial chamber", shape["special_info"]) is not None:
            filtered_shapes.append(shape)
    for shape in shapes:
        if shape["special_info"] == volution_max["special_info"]:
            filtered_shapes.append(shape)
    for shape in filtered_shapes:
        figure.draw(shape, color=fixed_color, trans=face_color)
    basic_img_before = figure.transfer_to_cv2(transparent=False)
    for shape in shapes:
        figure.draw(shape, color=fixed_color, trans=face_color)
    basic_img_after = figure.transfer_to_cv2()
    figure.close()
    return basic_img_before, basic_img_after, figure.volution_memory, figure.max_volution


def generate_septa(septas: list, debug=None) -> tuple:
    def fill_septas(xs, ys, centers, cv2_fig: np.ndarray, alpha_channel: np.ndarray):
        gray_cv2_fig = cv2.cvtColor(cv2_fig, cv2.COLOR_BGR2GRAY)
        for x, y, center in zip(xs, ys, centers):
            if center is None:
                continue
            cx = int(center[0] * figure.shape[0])
            cy = int((1 - center[1]) * figure.shape[1])
            # cv2_fig[cy][cx] = np.array([255,0,0])
            # cv2.imwrite("DEBUG_SPOT_MID.png", cv2_fig)
            history = []
            x = figure.shape[0] * x
            y = figure.shape[1] * (1 - y)
            for fx, fy in zip(x, y):
                step = 0
                cosine = (fx - cx) / np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
                sine = (fy - cy) / np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
                while True:
                    step += 1
                    dx, dy = int(cx + cosine * step), int(cy + sine * step)
                    if (dx, dy) in history:
                        continue
                    if (
                        (abs(dx - cx) >= 200 and abs(dy - cy) >= 200)
                        or dx < 0
                        or dx >= figure.shape[0]
                        or dy < 0
                        or dy >= figure.shape[1]
                    ):
                        break
                    # print(type(cv2_fig[dx][dy]),cv2_fig[dx][dy])
                    # print(dx,dy)
                    # assert 0
                    if gray_cv2_fig[dy][dx] < 127:
                        # print(cv2_fig[dx][dy])
                        break
                    # print(f"Converted {(dx,dy)} from {cv2_fig[dx][dy]} to 0")
                    alpha_channel[dy][dx] = 255
                    history.append((dx, dy))
        whiteboard = np.full(alpha_channel.shape, 255, dtype=alpha_channel.dtype)
        alpha_channel = np.where(gray_cv2_fig < 32, whiteboard, alpha_channel)
        return alpha_channel

    figure = Figure_Engine(xkcd=True)
    fixed_color = np.random.randn(3) / 10 + np.array((0.1, 0.1, 0.1))

    xs = []
    ys = []
    centers = []
    for septa in septas:
        pack = figure.draw(septa, color=fixed_color)
        if pack is None:
            continue
        else:
            x, y, center = pack
        xs.append(x)
        ys.append(y)
        centers.append(center)

    cv2_fig = figure.transfer_to_cv2()
    cv2.imwrite("DEBUG_HERE_1.png", cv2_fig)
    alpha_channel = cv2_fig[:, :, 3]
    # print(alpha_channel)
    cv2_fig = cv2_fig[:, :, 0:3]
    if debug is not None:
        cv2.imwrite(f"{debug}/test_septa_xkcd.jpg", cv2_fig)
    cv2_fig = Image.fromarray(cv2.cvtColor(cv2_fig, cv2.COLOR_BGR2RGB))
    cv2_fig = cv2_fig.convert("L")
    cv2_fig = cv2_fig.convert("RGB")
    cv2_fig = np.asarray(cv2_fig)
    cv2_fig = cv2.cvtColor(cv2_fig, cv2.COLOR_RGB2BGR)
    alpha_channel = fill_septas(xs, ys, centers, cv2_fig, alpha_channel)
    figure.close()
    cv2.imwrite("DEBUG_HERE_2.png", alpha_channel)
    return cv2_fig, alpha_channel


def post_processing(img: np.ndarray):
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    sharpened = cv2.addWeighted(img, 1, laplacian, 1, 0)
    return sharpened


def separate_axial_filling(axial_filling: list) -> tuple:
    main_af = []
    ext_af = []
    for af in axial_filling:
        if af["type"] == "main":
            main_af.append(af)
        elif af["type"] == "extension":
            ext_af.append(af)
    return main_af, ext_af

def generate_one_img(
    idx,
    sample,
    img_path: str,
    ref_path: str,
    ref_poles_pool: str,
    num_refs: int,
    keyword,
    best_ref: dict = {},
):
    debug_folder = keyword
    if not os.path.exists(f"{debug_folder}/DEBUG"):
        os.makedirs(f"{debug_folder}/DEBUG", exist_ok=True)
    basic_img, basic_img_after, volution_memory, max_volution = generate_basic_shape_separately(
        sample["shapes"], sample["numerical_info"]
    )
    basic_img_copy, basic_img_copy_2 = np.copy(basic_img), np.copy(basic_img)
    best_ref_poles = best_ref

    main_af, ext_af = separate_axial_filling(sample["axial_filling"])
    main_mask = generate_basic_mask(volution_memory, main_af, "axial_main", debug=f"{debug_folder}/DEBUG/{idx}")
    diffused_basic_img = diffuse(
        basic_img,
        main_mask,
        best_ref_poles,
        ref_path,
        num_refs,
        mode="axial_main",
        debug=f"{debug_folder}/DEBUG/{idx}_pre",
    )
    
    ext_mask = generate_basic_mask(volution_memory, ext_af, "axial_ext", debug=f"{debug_folder}/DEBUG/{idx}")
    diffused_basic_img_2 = diffuse(
        basic_img_copy,
        ext_mask,
        best_ref_poles,
        ref_path,
        num_refs,
        mode="axial_ext",
        debug=f"{debug_folder}/DEBUG/{idx}_pre",
    )
    ext_mask = cv2.cvtColor(ext_mask, cv2.COLOR_BGR2GRAY)
    ext_mask = ext_mask[..., np.newaxis]
    diffused_basic_img = np.where(ext_mask <= 128, diffused_basic_img_2, diffused_basic_img)
    
    diffused_basic_img[basic_img_after[:, :, 3] > 0] = basic_img_after[basic_img_after[:, :, 3] > 0][:, :3]

    # diffused_basic_img = diffuse(basic_img, basic_mask, best_ref_poles, ref_path, num_refs, mode = 'axial',debug=None)
    # septa_overlayer = generate_septa(sample["septa_folds"])
    # diffused_basic_img = np.minimum(diffused_basic_img,septa_overlayer)
    
    poles_mask = generate_basic_mask(
        volution_memory, sample["poles_folds"], "poles", debug=f"{debug_folder}/DEBUG/{idx}"
    )
    diffused_img = diffuse(
        basic_img_copy_2,
        poles_mask,
        best_ref_poles,
        ref_path,
        num_refs,
        mode="poles",
        debug=f"{debug_folder}/DEBUG/{idx}_post",
    )
    poles_mask = cv2.cvtColor(poles_mask, cv2.COLOR_BGR2GRAY)
    poles_mask = poles_mask[..., np.newaxis]
    diffused_img = np.where(poles_mask <= 128 , diffused_img, diffused_basic_img)
    
    diffused_img = redraw_basic_shapes(diffused_img, sample["shapes"])
    septa_overlayer, alpha_mask = generate_septa(sample["septa_folds"])

    alpha_mask = alpha_mask[..., np.newaxis]
    blended_img = np.where(alpha_mask == 255, septa_overlayer, diffused_img)

    blended_img = cv2.cvtColor(blended_img, cv2.COLOR_BGRA2GRAY)
    final_img = post_processing(blended_img)
    img_path = f"{keyword}/{img_path}"
    if not os.path.exists(f"{keyword}"):
        os.mkdir(f"{keyword}")
    cv2.imwrite(img_path, final_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=str)
    parser.add_argument("--best_match", type=str)
    parser.add_argument("--start_pos", type=int, default=0)
    parser.add_argument("--end_pos", type=int, default=None)
    parser.add_argument("--kwd", type=str)
    args = parser.parse_args()

    with open(args.rules, "r") as f:
        # "/home/nfs04/xingsy/geocap/dataset/rules_stage2_100k_part2.json"
        samples = json.load(f)
        # random.shuffle(samples)
        # samples=samples[:10]
        assert isinstance(samples, list)
    best_ref_list = []
    with open(args.best_match, "r") as f:
        # "stage2_100k_bestmatch_part2.json"
        best_ref_list = json.load(f)
    if args.end_pos is not None:
        samples = samples[args.start_pos : args.end_pos]
        best_ref_list = best_ref_list[args.start_pos : args.end_pos]
    else:
        samples = samples[args.start_pos :]
        best_ref_list = best_ref_list[args.start_pos :]

    for idx_sample, sample in enumerate(tqdm(samples)):
        # if idx_sample not in [7,8,9]:
        #     continue
        generate_one_img(
            idx_sample + args.start_pos,
            sample,
            f"{idx_sample+args.start_pos:08d}.jpg",
            ref_path="fos_data/reference_aug_14th",
            ref_poles_pool="pics_8xx/",
            num_refs=10,  # some numberd
            keyword=args.kwd,  # "100k_finetune_internvl_part2"
            best_ref=best_ref_list[idx_sample],
        )


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    main()

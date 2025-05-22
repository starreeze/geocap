import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.append(".")
sys.path.append("..")
from run_gradio3_demo import crop_padding_and_resize, inference_single_image


def processRefImage(ref_image):
    height, width = ref_image.shape[:2]

    # 计算中间位置的x坐标
    center_x = width // 2

    # 定义线的起点和终点坐标
    # 起点: (center_x, 0) - 图像顶部
    # 终点: (center_x, height-1) - 图像底部
    start_point = (center_x, 0)
    end_point = (center_x, height - 1)

    # 定义线的颜色 (BGR格式)，默认为红色
    color = (255, 255, 255)

    # 定义线的粗细，默认为1像素
    thickness = 50

    # 在图像上绘制线
    image_with_line = cv2.line(ref_image, start_point, end_point, color, thickness)

    return image_with_line


def get_random_match(dir: str):
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    lucky_guy = random.choice(files)
    return lucky_guy


def diffuse(
    img: np.ndarray,
    mask: np.ndarray,
    best_ref_poles: dict,
    ref_path: str,
    num_refs: int,
    mode: str,
    debug=None,
) -> np.ndarray:

    def get_random_ref_image(ref_path: list) -> np.ndarray:
        ref_image = cv2.imread(random.choice(ref_path))
        return ref_image

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_img = pil_img.convert("L").convert("RGB")
    img = np.asarray(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pil_mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    pil_mask = pil_mask.convert("L")
    mask = np.asarray(pil_mask)
    mask = 255 - mask
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)

    if mode == "axial":
        ref_paths = [os.path.join(ref_path, f"ref_axial_{x:08}.png") for x in range(num_refs)]
        ref_image = get_random_ref_image(ref_paths)
    elif mode == "poles":
        # ref_paths = [os.path.join(ref_path, f"ref_{x:08}.png") for x in range(num_refs)]
        # ref_image = get_best_match(img, ref_paths)
        ref_image = cv2.imread(best_ref_poles["best_ref"], cv2.IMREAD_UNCHANGED)
        transparent_mask = ref_image[:, :, 3] == 0  # alpha通道为0的像素
        ref_image[transparent_mask] = [255, 255, 255, 255]  # 一次性设置RGBA
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGBA2BGR)
        target_width = abs(best_ref_poles["bbox"][0][0] - best_ref_poles["bbox"][1][0])
        target_height = abs(best_ref_poles["bbox"][0][1] - best_ref_poles["bbox"][1][1])
        original_height, original_width = ref_image.shape[:2]
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height

        # 计算新尺寸
        if original_ratio < target_ratio:
            # 原始图像更"瘦高"，需要放大宽度
            new_width = int(original_height * target_ratio)
            new_height = original_height
        else:
            # 原始图像更"矮胖"，需要放大高度
            new_width = original_width
            new_height = int(original_width / target_ratio)

        # 使用INTER_CUBIC插值进行放大（保持高质量）
        ref_image = cv2.resize(ref_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        ref_image = processRefImage(ref_image)

    if debug is not None:
        cv2.imwrite(f"{debug}_Image.png", img)
        cv2.imwrite(f"{debug}_Reference.png", ref_image)

    synthesis, depth_pred = inference_single_image(
        ref_image.copy(), img.copy(), mask.copy(), ddim_steps=120, scale=10, seed=0, enable_shape_control=True
    )

    synthesis = crop_padding_and_resize(img, synthesis)
    if debug is not None:
        cv2.imwrite(f"{debug}_SYNTH_1.png", synthesis)

    return synthesis.astype(np.uint8)

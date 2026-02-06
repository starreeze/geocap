"""
获取化石图像的轮廓并计算轮廓内的面积
"""

import json
from pathlib import Path

import cv2
import numpy as np


def get_fossil_contour(img_path: str) -> np.ndarray | None:
    """
    获取化石图像的外轮廓（基于透明通道）

    Parameters:
        img_path: 图像路径（需要带透明通道，非透明像素为化石区域）

    Returns:
        轮廓点数组，如果未找到则返回 None
    """
    # 读取带透明通道的图像
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # 检查是否有透明通道
    if img.shape[2] != 4:
        return None

    # 使用 alpha 通道作为 mask
    alpha = img[:, :, 3]

    # 查找外轮廓
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return None

    # 返回面积最大的轮廓
    return max(contours, key=cv2.contourArea)


def get_fossil_area(
    img_path: str, pixel_per_mm: float | None = None
) -> tuple[np.ndarray | None, float, float | None]:
    """
    获取化石轮廓及其面积

    Parameters:
        img_path: 图像路径
        pixel_per_mm: 每像素对应的毫米数（比例尺），如果提供则计算实际面积

    Returns:
        (轮廓点数组, 像素面积, 实际面积mm²)
        如果未找到轮廓则返回 (None, 0.0, None)
        如果未提供比例尺则实际面积为 None
    """
    contour = get_fossil_contour(img_path)
    if contour is None:
        return None, 0.0, None

    pixel_area = cv2.contourArea(contour)

    # 计算实际面积 (mm²)
    real_area = None
    if pixel_per_mm is not None:
        # pixel_per_mm 是每像素对应的毫米数
        # 面积 = 像素数 * (mm/pixel)²
        real_area = pixel_area * (pixel_per_mm**2)

    return contour, pixel_area, real_area


def load_scale_from_data_json(img_name: str, data_json_path: str = "dataset/data.json") -> float | None:
    """
    从 data.json 中加载图像的比例尺

    Parameters:
        img_name: 图像文件名（如 "Chusenella_absidata_1_1.png"）
        data_json_path: data.json 的路径

    Returns:
        pixel/mm 值，如果未找到则返回 None
    """
    with open(data_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for species_key, species_data in data.items():
        for img_info in species_data.get("images", []):
            if img_info.get("image") == img_name:
                return img_info.get("pixel/mm")

    return None


if __name__ == "__main__":
    import argparse
    import csv

    parser = argparse.ArgumentParser(
        description="计算化石图像的轮廓面积（基于透明通道）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python get_full_area.py ./images ./output/area.csv
  python get_full_area.py ./images ./output/area.csv --data-json ./dataset/data.json

输出格式 (CSV):
  image,pixel_area,real_area_mm2
  image_name.png,123456.0,14.93
  ...

注意:
  - 图像必须带有透明通道 (RGBA)，非透明像素区域被视为化石区域
  - 比例尺从 data.json 中根据文件名自动查找
  - 如果找不到比例尺，real_area_mm2 将为空
""",
    )
    parser.add_argument("image_dir", type=str, help="输入图像目录路径")
    parser.add_argument("output_csv", type=str, help="输出 CSV 文件路径")
    parser.add_argument(
        "--data-json",
        type=str,
        default="dataset/data.json",
        help="包含比例尺信息的 data.json 路径 (默认: dataset/data.json)",
    )

    args = parser.parse_args()

    img_dir = Path(args.image_dir)
    output_csv = args.output_csv
    data_json_path = args.data_json

    if not img_dir.exists():
        print(f"错误: 目录不存在: {img_dir}")
        exit(1)

    # 获取所有图片
    image_files = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    print(f"找到 {len(image_files)} 张图片")

    results = []
    for i, img_path in enumerate(image_files):
        img_name = img_path.name
        print(f"[{i+1}/{len(image_files)}] {img_name}")

        # 从 data.json 加载比例尺
        pixel_per_mm = load_scale_from_data_json(img_name, data_json_path)

        contour, pixel_area, real_area = get_fossil_area(str(img_path), pixel_per_mm)
        if contour is None:
            print("  跳过: 未找到轮廓")
            continue

        results.append(
            {
                "image": img_name,
                "pixel_area": pixel_area,
                "real_area_mm2": real_area if real_area is not None else "",
            }
        )

    # 保存结果
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "pixel_area", "real_area_mm2"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n结果已保存到: {output_csv}")
    print(f"成功处理: {len(results)} 张图片")

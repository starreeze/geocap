# flake8: noqa
import argparse
import json
import os
import random
import re
from io import BytesIO
from typing import Any

import cv2
import matplotlib.patches as pch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.interpolate import RBFInterpolator, interp1d

# from shape_filter import getMostSimilarImages
from tqdm import tqdm

# from concurrent.futures import ThreadPoolExecutor
# import concurrent


class Figure_Engine:
    # No longer randomization is available: this will be automatically on.
    # xkcd will be on.
    def __init__(
        self,
        size: "tuple[float, float]" = (12.8, 12.8),
        dpi: int = 100,
        max_volution: int = 0,
        center: tuple = (0.5, 0.5),
    ) -> None:
        self.image = plt.figure(figsize=size, dpi=dpi)
        self.shape = (int(size[0] * dpi), int(size[1] * dpi))
        self.ax = self.image.add_subplot()
        plt.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.volution_memory = {}
        self.max_volution = max_volution
        self.center = center

    def draw(self, shape, width=None, color=None, trans=None):
        width, color, trans = self.__get_essential_info(shape, width, color, trans)
        try:
            index = self.__special_info_validator(shape["special_info"])
        except:
            index = None
        assert index == None or (-1 <= index and index <= 20)
        if shape["type"] == "fusiform_1":
            x_offset = shape["x_offset"]
            y_offset = shape["y_offset"]
            fc = shape["focal_length"]
            eps, ome, phi = shape["sin_params"]
            x_start = shape["x_start"]
            x_end = shape["x_end"]
            y_sim = shape["y_symmetric_axis"]

            self.__handle_fusiform_1(
                fc, x_offset, y_offset, eps, ome, phi, x_start, x_end, y_sim, color, trans, width, index
            )

        elif shape["type"] == "fusiform_2":
            x_offset = shape["x_offset"]
            y_offset = shape["y_offset"]
            fc = shape["focal_length"]
            eps, ome, phi = shape["sin_params"]
            power = shape["power"]
            x_start = shape["x_start"]
            x_end = shape["x_end"]

            self.__handle_fusiform_2(
                focal_length=fc,
                x_offset=x_offset,
                y_offset=y_offset,
                power=power,
                eps=eps,
                omega=ome,
                phi=phi,
                x_start=x_start,
                x_end=x_end,
                color=color,
                trans=trans,
                line_width=width,
                index=index,
            )

        elif shape["type"] == "ellipse":
            ellipse_x, ellipse_y = shape["center"]
            major = shape["major_axis"]
            minor = shape["minor_axis"]
            alpha = shape["rotation"] * 180 / np.pi

            self.__handle_ellipse(ellipse_x, ellipse_y, major, minor, alpha, width, color, trans, index)

        elif shape["type"] == "polygon":
            points: list = shape["points"]
            assert len(points) >= 3, "There should be more than 3 points within a polygon."

            self.__handle_polygon(points, width, color, trans)

        elif shape["type"] == "spindle":
            center_x, center_y = shape["center"]
            major = shape["major_axis"]
            minor = shape["minor_axis"]

            self.__handle_spindle(center_x, center_y, major, minor, width, color, index)

        elif shape["type"] == "curves":
            curves = shape["control_points"]
            if isinstance(curves[0][0], float):
                curves = [curves]
            curve_buf_x = []
            curve_buf_y = []
            for curve in curves:
                x, y = self.__handle_curve(curve, width, color, index)
                curve_buf_x.append(x)
                curve_buf_y.append(y)

            curve_buf_x = [item for x in curve_buf_x for item in x]
            curve_buf_y = [item for y in curve_buf_y for item in y]
            """
                debug_fig = plt.figure()
                ax = debug_fig.add_subplot().scatter(curve_buf_x, curve_buf_y, s=1, marker='x')
                debug_fig.savefig(f"scatter_points{index}.png")
                #"""
            self.__keep_memory(index, curve_buf_x, curve_buf_y)

    def __handle_ellipse(
        self,
        ellipse_x: float,
        ellipse_y: float,
        major: float,
        minor: float,
        alpha: float,
        line_width: int,
        color: Any,
        transparency: tuple = (0, 0, 0, 0),
        index=None,
    ):
        if major < minor:
            raise ValueError("The major axis is smaller than the minor axis, which is incorrect.")
        self.ax.add_patch(
            pch.Ellipse(
                (ellipse_x, ellipse_y),
                major,
                minor,
                angle=alpha,
                edgecolor=color,
                facecolor=transparency,
                linewidth=line_width * (self.shape[0] / 640),
            )
        )
        # Proloculus version
        self.__keep_memory(index, ellipse_x, ellipse_y)

    def __handle_polygon(self, points: list, line_width: int, color: Any, trans: tuple = (0, 0, 0, 0)):
        self.ax.add_patch(
            pch.Polygon(
                points,
                closed=True,
                edgecolor=color,
                linewidth=line_width * (self.shape[0] / 640),
                facecolor=trans,
            )
        )

    def __handle_spindle(
        self,
        center_x: float,
        center_y: float,
        major_axis: float,
        minor_axis: float,
        line_width: int,
        color: Any,
        index=None,
    ):
        theta = np.arange(0, 2 * 3.1416, 0.01)
        a = major_axis / 2
        b = minor_axis / 2

        rho = np.sqrt(1 / (np.cos(theta) ** 2 / a**2 + np.sin(theta) ** 2 / b**2))
        rho1 = np.sqrt(np.abs((a / 10) ** 2 * np.sin(2 * (theta + 1.5708))))
        rho2 = np.sqrt(np.abs((a / 10) ** 2 * np.sin(2 * theta)))
        rho_list = rho - rho1 - rho2  # shift on pi/4s.

        rho_ru = rho_list[np.where((theta < 3.1416 * 0.35))]
        theta_ru = theta[np.where((theta < 3.1416 * 0.35))]
        x_ru = (rho_ru) * np.cos(theta_ru) + center_x
        y_ru = (rho_ru) * np.sin(theta_ru) + center_y

        rho_rd = rho_list[np.where((theta > 3.1416 * 1.65))]
        theta_rd = theta[np.where((theta > 3.1416 * 1.65))]
        x_rd = (rho_rd) * np.cos(theta_rd) + center_x
        y_rd = (rho_rd) * np.sin(theta_rd) + center_y

        rho_l = rho_list[np.where((theta > 3.1416 * 0.65) & (theta < 3.1416 * 1.35))]
        theta_l = theta[np.where((theta > 3.1416 * 0.65) & (theta < 3.1416 * 1.35))]
        x_l = (rho_l) * np.cos(theta_l) + center_x
        y_l = (rho_l) * np.sin(theta_l) + center_y

        x_mu = np.linspace(x_ru[-1], x_l[0], num=5)
        y_mu = np.linspace(y_ru[-1], y_l[0], num=5)

        x_md = np.linspace(x_l[-1], x_rd[0], num=5)
        y_md = np.linspace(y_l[-1], y_rd[0], num=5)

        x = np.concat((x_ru, x_mu, x_l, x_md, x_rd), axis=None)
        y = np.concat((y_ru, y_mu, y_l, y_md, y_rd), axis=None)

        self.ax.plot(x, y, color=color, linewidth=line_width * (self.shape[0] / 640))

        self.__keep_memory(index, x, y)

    def __handle_fusiform_1(
        self,
        focal_length,
        x_offset,
        y_offset,
        eps,
        omega,
        phi,
        x_start,
        x_end,
        y_sim,
        color,
        trans,
        line_width,
        index,
    ):
        def f(x):
            return 4 * focal_length * (x - x_offset) ** 2 + y_offset + eps * np.sin(omega * x + phi)

        x = np.linspace(x_start, x_end, 1000)
        y1 = f(x)
        y2 = 2 * y_sim - y1

        for i in range(len(x)):
            self.ax.plot((x[i], x[i]), (y1[i], y2[i]), linewidth=1, color=trans)

        self.ax.plot(x, y1, x, y2, linewidth=line_width * (self.shape[0] / 640), color=color)

        self.__keep_memory(index, np.concatenate([x, x]), np.concatenate([y1, y2]))

    def __handle_fusiform_2(
        self,
        focal_length,
        x_offset,
        y_offset,
        power,
        eps,
        omega,
        phi,
        x_start,
        x_end,
        color,
        trans,
        line_width,
        index,
    ):
        x = np.linspace(x_start, x_end, 1000)
        x_left = x[:500]
        sin_wave = eps * np.sin(omega * (x - x_start) + phi)
        y_left = (np.abs(x_left - x_offset) / (4 * focal_length)) ** (1 / power) + y_offset
        y_right = np.flip(y_left)  # 得到开口向左的上半部分
        y1 = np.concatenate([y_left, y_right]) + sin_wave
        y2 = 2 * y_offset - y1  # 得到整个纺锤形的下半部分
        # fix the side-wise crack
        x_start_lst = np.array([x_start for _ in range(50)])
        x_end_lst = np.array([x_end for _ in range(50)])
        y_start_lst = np.linspace(y1[0], y2[0], 50)
        y_end_lst = np.linspace(y1[-1], y2[-1], 50)
        for i in range(len(x)):
            self.ax.plot(
                (x[i], x[i]), (y1[i], y2[i]), linewidth=line_width * (self.shape[0] / 640), color=trans
            )
        self.ax.plot(
            x,
            y1,
            x,
            y2,
            x_start_lst,
            y_start_lst,
            x_end_lst,
            y_end_lst,
            linewidth=line_width * (self.shape[0] / 640),
            color=color,
        )

        self.__keep_memory(
            index,
            np.concatenate([x, x, x_start_lst, x_end_lst]),
            np.concatenate([y1, y2, y_start_lst, y_end_lst]),
        )

    def __handle_curve(self, control_points, width: int = 5, color=(0, 0, 0, 1), index=None):
        curve_points = []
        t_values = np.linspace(0, 1, 600)
        for t in t_values:
            one_minus_t = 1 - t
            point = (
                one_minus_t**3 * np.array(control_points[0])
                + 3 * one_minus_t**2 * t * np.array(control_points[1])
                + 3 * one_minus_t * t**2 * np.array(control_points[2])
                + t**3 * np.array(control_points[3])
            )
            curve_points.append(tuple(point))
        curve_points = np.array(curve_points)
        self.ax.plot(
            curve_points[:, 0], curve_points[:, 1], linewidth=width * (self.shape[0] / 640), color=color
        )

        return curve_points[:, 0], curve_points[:, 1]

    def __get_essential_info(self, shape, width, color, trans):
        if width is None:
            width = self.__get_width(shape)
        if color is None:
            color = self.__get_color(shape)
        color = np.clip(color, 0, 1)
        if trans is None:
            trans = self.__get_transparency(shape)
        return width, color, trans

    def __get_width(self, shape):
        try:
            return shape["width"]
        except:
            return 3

    def __get_color(self, shape):
        try:
            return shape["color"]
        except:
            return (random.random(), random.random(), random.random())

    def __get_transparency(self, shape):
        try:
            if shape["fill_mode"] == "no":
                trans = (0, 0, 0, 0)
            elif shape["fill_mode"] == "white":
                trans = (1, 1, 1, 1)
            elif shape["fill_mode"] == "black":
                trans = (0, 0, 0, 1)
        except:
            trans = (0, 0, 0, 0)  # no
        return trans

    def transfer_to_cv2_wrapper(self):
        buf2 = BytesIO()
        self.image.savefig(buf2, transparent=True, format="png")
        buf2.seek(0)

        query_img_array = np.frombuffer(buf2.getvalue(), dtype=np.uint8)
        query_img = cv2.imdecode(query_img_array, cv2.IMREAD_UNCHANGED)

        return query_img

    def transfer_to_cv2(self):
        buf = BytesIO()
        # buf2=BytesIO()
        self.image.savefig(buf, format="png", facecolor="#FFFFFF")
        # self.image.savefig(buf2,transparent=True,format="png")
        buf.seek(0)
        # buf2.seek(0)

        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        # query_img_array=np.frombuffer(buf2.getvalue(),dtype=np.uint8)
        # query_img=cv2.imdecode(query_img_array,cv2.IMREAD_UNCHANGED)

        return img

    def __special_info_validator(self, special_info):
        for i in range(20):  # How a fusi has volutions over 20 layers?
            if special_info == f"volution {i}":
                return i
            if special_info == "initial chamber":
                return -1
        # print(special_info)
        return None

    def __keep_memory(self, index, x, y):
        assert (isinstance(x, float) and isinstance(y, float)) or (len(x) == len(y))
        # Suppose the center is the same
        if index != None and index >= 0:
            angle_value = []
            for x0, y0 in zip(x, y):
                x0 -= self.center[0]
                y0 -= self.center[1]
                if x0 == 0:
                    if y0 > 0:
                        angle_value.append(np.pi / 2)
                    else:
                        angle_value.append(3 * np.pi / 2)
                else:
                    if x0 > 0 and y0 >= 0:
                        this_angle = np.arctan(y0 / x0)
                    elif x0 > 0 and y0 < 0:
                        this_angle = 2 * np.pi + np.arctan(y0 / x0)
                    elif x0 < 0:
                        this_angle = np.pi + np.arctan(y0 / x0)
                    angle_value.append(this_angle)
            angle_value = np.array(angle_value)
            indices = np.argsort(angle_value)
            x = np.array(x)[indices]
            y = np.array(y)[indices]
            angle_value = angle_value[indices]
            self.volution_memory[index] = (x, y, angle_value)
        if index == -1:
            self.volution_memory[index] = self.center

    def close(self):
        plt.close(self.image)


def generate_basic_shape_wrapper(data_dict):
    shapes = data_dict["shapes"]
    ni = data_dict["ni"]
    figure = Figure_Engine(max_volution=int(ni["num_volutions"]), center=ni["center"])
    fixed_color = np.random.randn(3) / 6 + np.array((0.1, 0.1, 0.1))
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
    for shape in shapes:
        if shape["special_info"] == volution_max["special_info"]:
            filtered_shapes.append(shape)
    for shape in filtered_shapes:
        figure.draw(shape)
    query_img = figure.transfer_to_cv2_wrapper()
    figure.close()
    return query_img


def generate_basic_shape_separately(shapes: list, ni: dict):
    figure = Figure_Engine(max_volution=int(ni["num_volutions"]), center=ni["center"])
    fixed_color = np.random.randn(3) / 6 + np.array((0.1, 0.1, 0.1))
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
        figure.draw(shape)
    basic_img_before = figure.transfer_to_cv2()
    for shape in shapes:
        figure.draw(shape)
    basic_img_after = figure.transfer_to_cv2_wrapper()
    figure.close()
    return basic_img_before, basic_img_after, figure.volution_memory, figure.max_volution


def get_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    return contour.reshape(-1, 2)


def match_point_count(src_pts, dst_pts):
    # 确保两个轮廓点数一致：插值处理
    n_src, n_dst = len(src_pts), len(dst_pts)
    if n_src == n_dst:
        return src_pts, dst_pts

    # 较少点的作为目标数量
    target_n = min(n_src, n_dst)

    def interpolate(pts, new_n):
        x = pts[:, 0]
        y = pts[:, 1]
        t = np.linspace(0, 1, len(pts))
        t_new = np.linspace(0, 1, new_n)

        fx = interp1d(t, x, kind="linear")
        fy = interp1d(t, y, kind="linear")

        new_x = fx(t_new)
        new_y = fy(t_new)
        return np.column_stack((new_x, new_y))

    if n_src > n_dst:
        src_pts = interpolate(src_pts, target_n)
        dst_pts = dst_pts
    else:
        dst_pts = interpolate(dst_pts, target_n)
        src_pts = src_pts

    return src_pts.astype(np.float32), dst_pts.astype(np.float32)


def apply_tps_warp(img_src, img_dst, src_pts, dst_pts):
    # 构造TPS变换器
    tps = RBFInterpolator(src_pts, dst_pts, kernel="thin_plate_spline", neighbors=100)

    h, w = img_dst.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    warped_points = tps(grid_points)
    map_xy = warped_points.reshape((h, w, 2)).astype(np.float32)
    # map_xy[..., 0] = np.clip(map_xy[..., 0], 0, img_src.shape[1] - 1)
    # map_xy[..., 1] = np.clip(map_xy[..., 1], 0, img_src.shape[0] - 1)

    # 应用变换
    warped_img = cv2.remap(
        img_src, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    return warped_img


def paste_masked_region(image1, image2, mask):
    """
    将 image1 中 mask 区域的内容粘贴到 image2 的相同位置上
    :param image1: 源图像 (BGR格式)
    :param image2: 目标图像 (BGR格式)
    :param mask: 单通道二值 mask 图像 (0 或 255)
    :return: 融合后的图像
    """
    # 确保 mask 是单通道的
    if len(mask.shape) != 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 创建 mask 的布尔掩码
    mask_bool = mask.astype(bool)

    # 创建输出图像，初始化为 image2
    output = image2.copy()

    # 替换 image2 中 mask 区域为 image1 的内容
    output[mask_bool] = image1[mask_bool]

    return output


def resample_contour_uniform(contour, target_points):
    # 将轮廓点展平
    points = contour.squeeze()
    if len(contour) == target_points:
        return contour

    # 计算轮廓周长
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    cum_dist = np.insert(np.cumsum(distances), 0, 0)
    total_length = cum_dist[-1]

    # 计算新的采样点位置
    sample_distances = np.linspace(0, total_length, target_points)

    # 线性插值获取新点
    new_points = []
    for dist in sample_distances:
        idx = np.searchsorted(cum_dist, dist) - 1
        idx = max(0, min(idx, len(points) - 2))

        # 计算插值比例
        ratio = (dist - cum_dist[idx]) / (cum_dist[idx + 1] - cum_dist[idx])

        # 线性插值
        new_point = points[idx] + ratio * (points[idx + 1] - points[idx])
        new_points.append(new_point)

    return np.array(new_points).reshape(-1, 1, 2).astype(np.int32)


def tpsAxialFilling(base_image, axial_mask_left, axial_mask_right, ref_img_name, debug=None):
    contour_left = get_contour(255 - axial_mask_left)
    contour_right = get_contour(255 - axial_mask_right)
    contour_ref_left = get_contour(cv2.imread(f"axial_fillings/masks/{ref_img_name}-axial_filling_left.png"))
    contour_ref_right = get_contour(
        cv2.imread(f"axial_fillings/masks/{ref_img_name}-axial_filling_right.png")
    )

    ref_img = cv2.imread(f"axial_fillings/{ref_img_name}.png")

    # 匹配点数
    # num_points_left = min(len(contour_left), len(contour_ref_left))
    # contour_left = resample_contour_uniform(contour_left,num_points_left)
    # contour_ref_left = resample_contour_uniform(contour_ref_left,num_points_left)

    # num_points_right = min(len(contour_right), len(contour_ref_right))
    # contour_right = resample_contour_uniform(contour_right,num_points_right)
    # contour_ref_right = resample_contour_uniform(contour_ref_right,num_points_right)

    contour_left, contour_ref_left = match_point_count(contour_left, contour_ref_left)
    contour_right, contour_ref_right = match_point_count(contour_right, contour_ref_right)

    warped_left = apply_tps_warp(ref_img, axial_mask_left, contour_ref_left, contour_left)
    warped_right = apply_tps_warp(ref_img, axial_mask_right, contour_ref_right, contour_right)

    paste1 = paste_masked_region(warped_left, base_image, 255 - axial_mask_left)
    paste2 = paste_masked_region(warped_right, paste1, axial_mask_right)

    if debug != None:
        cv2.imwrite(f"{debug}_axial_left.png", paste1)
        cv2.imwrite(f"{debug}_axial_right.png", paste2)

    return paste2


def generate_basic_shape(shapes: list, ni: dict) -> tuple:
    figure = Figure_Engine(max_volution=int(ni["num_volutions"]), center=ni["center"])
    fixed_color = np.random.randn(3) / 6 + np.array((0.1, 0.1, 0.1))
    for shape in shapes:
        figure.draw(shape)
    basic_img = figure.transfer_to_cv2()
    figure.close()
    return basic_img, figure.volution_memory, figure.max_volution


def generate_basic_mask_separately(volution_memory: dict, filling: list, debug=None):
    from bisect import bisect

    mask_left = plt.figure(figsize=(12.8, 12.8), dpi=100)
    ax1 = mask_left.add_subplot()
    plt.subplots_adjust(0, 0, 1, 1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    mask_right = plt.figure(figsize=(12.8, 12.8), dpi=100)
    ax2 = mask_right.add_subplot()
    plt.subplots_adjust(0, 0, 1, 1)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    # volution_memory structure:
    # -1 -> initial chamber, (x,y) as the middle point
    # other numbers -> volutions, (x, y, angle) as the points of the volutions, sorted by angle[0:2pi~6.28]
    current_mask = mask_left
    for fill in filling:  # experiment suggests that diffusing these together works better
        start_angle = fill["start_angle"]
        current_ax = ax1 if start_angle > (np.pi / 2) else ax2
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
                current_ax.plot(
                    [start_volution[0][index_at_start], end_volution[0][index_at_end]],  # x
                    [start_volution[1][index_at_start], end_volution[1][index_at_end]],  # y
                    color="black",
                    linewidth=5,
                )
            except IndexError:
                current_ax.plot(
                    [start_volution[0][index_at_start - 1], end_volution[0][index_at_end - 1]],  # x
                    [start_volution[1][index_at_start - 1], end_volution[1][index_at_end - 1]],  # y
                    color="black",
                    linewidth=5,
                )
    if debug != None:
        mask_left.savefig(f"{debug}_Mask_original_left.png")
        mask_right.savefig(f"{debug}_Mask_original_right.png")
    buf = BytesIO()
    mask_left.savefig(buf, format="png", facecolor="#FFFFFF")
    buf.seek(0)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img1 = cv2.imdecode(img_array, 1)

    plt.close(mask_left)

    buf = BytesIO()
    mask_right.savefig(buf, format="png", facecolor="#FFFFFF")
    buf.seek(0)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img2 = cv2.imdecode(img_array, 1)

    plt.close(mask_right)

    height, width = img1.shape[:2]

    color = (255, 255, 255)

    rect_thickness = 5

    cv2.rectangle(img1, (0, 0), (width, rect_thickness), color, -1)
    cv2.rectangle(img1, (0, height - rect_thickness), (width, height), color, -1)
    cv2.rectangle(img1, (0, 0), (rect_thickness, height), color, -1)
    cv2.rectangle(img1, (width - rect_thickness, 0), (width, height), color, -1)

    cv2.rectangle(img2, (0, 0), (width, rect_thickness), color, -1)
    cv2.rectangle(img2, (0, height - rect_thickness), (width, height), color, -1)
    cv2.rectangle(img2, (0, 0), (rect_thickness, height), color, -1)
    cv2.rectangle(img2, (width - rect_thickness, 0), (width, height), color, -1)

    return img1, img2


def generate_basic_mask(volution_memory: dict, filling: list, debug=None) -> np.ndarray:
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
    if debug != None:
        mask.savefig(f"{debug}_Mask_original.png")
    buf = BytesIO()
    mask.savefig(buf, format="png")
    buf.seek(0)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    plt.close(mask)

    return img


def get_hu_moments(img: np.ndarray):
    edges = cv2.Canny(img, 75, 140)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[0])
    hu_moments = cv2.HuMoments(moments)  # Hu Moments for current picture
    return hu_moments


# def get_best_match_wrapper(data_dict):
#    return getMostSimilarImages(data_dict["pool_prefix"],data_dict["img"],data_dict["bbox"],n=16,max_sample=99999,batchsize=128,debug=False,multi_process=False,tmp_dir=data_dict["tmp_dir"])[0][0]

# def get_best_match(img, pool_prefix:str, bbox, tmp_dir):
# hu1 = get_hu_moments(img)
# picture_hus = [get_hu_moments(cv2.imread(filename)) for filename in ref_paths]
# distances = [hu1**2 + hu**2 for hu in picture_hus] # Euclid
# minimum_index = np.argmin(distances)
# return cv2.imread(ref_paths[minimum_index])
#    return getMostSimilarImages(pool_prefix,img,bbox,n=16,max_sample=99999,batchsize=128,debug=False,multi_process=True, tmp_dir=tmp_dir)[0][0]


def get_random_match(dir: str):
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    lucky_guy = random.choice(files)
    return lucky_guy


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


def diffuse(
    img: np.ndarray,
    mask: np.ndarray,
    best_ref_poles: dict,
    ref_path: str,
    num_refs: int,
    mode: str,
    debug=None,
    sample=None,
) -> np.ndarray:

    def get_random_ref_image(ref_path: list) -> np.ndarray:
        ref_image = cv2.imread(random.choice(ref_path))
        return ref_image

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.convert("L")
    img = img.convert("RGB")
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    mask = mask.convert("L")
    mask = np.asarray(mask)
    mask = 255 - mask
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)

    if mode == "axial":
        ref_paths = [os.path.join(ref_path, f"ref_axial_{x:08}.png") for x in range(num_refs)]
        ref_image = get_random_ref_image(ref_paths)
    elif mode == "poles":
        # ref_paths = [os.path.join(ref_path, f"ref_{x:08}.png") for x in range(num_refs)]
        # ref_image = get_best_match(img, ref_paths)
        ref_image = cv2.imread(best_ref_poles["best_ref"], cv2.IMREAD_UNCHANGED)
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
        # query_image_outer = generate_basic_shape_wrapper({
        #     "shapes":sample["shapes"], "ni":sample["numerical_info"]
        # })
        # ref_image = getTPSReference(query_image_outer,ref_image)
        transparent_mask = ref_image[:, :, 3] == 0  # alpha通道为0的像素
        ref_image[transparent_mask] = [255, 255, 255, 255]  # 一次性设置RGBA
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGBA2BGR)
        ref_image = processRefImage(ref_image)

    if debug != None:
        cv2.imwrite(f"{debug}_Image.png", img)
        cv2.imwrite(f"{debug}_Reference.png", ref_image)

    synthesis, depth_pred = inference_single_image(
        ref_image.copy(), img.copy(), mask.copy(), ddim_steps=120, scale=10, seed=0, enable_shape_control=True
    )

    synthesis = crop_padding_and_resize(img, synthesis)
    if debug != None:
        cv2.imwrite(f"{debug}_SYNTH_1.png", synthesis)

    return synthesis.astype(np.uint8)


def generate_septa(septas: list, debug=None) -> np.ndarray:
    figure = Figure_Engine()
    fixed_color = np.random.randn(3) / 6 + np.array((0.1, 0.1, 0.1))
    for septa in septas:
        figure.draw(septa, color=fixed_color)
    cv2_fig = figure.transfer_to_cv2()
    if debug != None:
        cv2.imwrite(f"test_septa_xkcd.jpg", cv2_fig)
    cv2_fig = Image.fromarray(cv2.cvtColor(cv2_fig, cv2.COLOR_BGR2RGB))
    cv2_fig = cv2_fig.convert("L")
    cv2_fig = cv2_fig.convert("RGB")
    cv2_fig = np.asarray(cv2_fig)
    cv2_fig = cv2.cvtColor(cv2_fig, cv2.COLOR_RGB2BGR)
    figure.close()
    return cv2_fig


def generate_one_img(
    idx, sample, img_path: str, ref_path: str, ref_poles_pool: str, num_refs: int, keyword, best_ref=None
):
    debug_folder = keyword
    if not os.path.exists(f"{debug_folder}/DEBUG"):
        os.makedirs(f"{debug_folder}/DEBUG", exist_ok=True)
    basic_img_before, basic_img_after, volution_memory, max_volution = generate_basic_shape_separately(
        sample["shapes"], sample["numerical_info"]
    )
    if best_ref is None:
        try:
            # best_ref_poles=get_best_match(query_img,ref_poles_pool,sample["numerical_info"]["fossil_bbox"])
            # print(best_ref_poles)
            best_ref_poles = "pics/Neoschwagerina_takagamiensis_1_2.png"
        except Exception as e:
            print(e)
            best_ref_poles = "pics/Neoschwagerina_takagamiensis_1_2.png"
    else:
        best_ref_poles = best_ref
    basic_mask = generate_basic_mask(volution_memory, sample["axial_filling"])
    # basic_mask_left,basic_mask_right=generate_basic_mask_separately(volution_memory, sample["axial_filling"],debug=f'{debug_folder}/DEBUG/{idx}_post')

    diffused_basic_img = diffuse(
        basic_img_before, basic_mask, best_ref_poles, ref_path, num_refs, mode="axial", debug=None
    )
    # diffused_basic_img = tpsAxialFilling(basic_img_before,basic_mask_left,basic_mask_right,"Chusenella_absidata_1_4", debug=f'{debug_folder}/DEBUG/{idx}_post')

    diffused_basic_img[basic_img_after[:, :, 3] > 0] = basic_img_after[basic_img_after[:, :, 3] > 0][:, :3]
    # septa_overlayer = generate_septa(sample["septa_folds"])
    # diffused_basic_img = np.minimum(diffused_basic_img,septa_overlayer)
    poles_mask = generate_basic_mask(volution_memory, sample["poles_folds"], debug=None)
    # diffused_img = diffuse(diffused_basic_img, poles_mask, best_ref_poles, ref_path, num_refs, mode = 'poles',debug=f'{debug_folder}/DEBUG/{idx}_post')
    diffused_img = diffuse(
        diffused_basic_img,
        poles_mask,
        best_ref_poles,
        ref_path,
        num_refs,
        mode="poles",
        debug=None,
        sample=sample,
    )
    septa_overlayer = generate_septa(sample["septa_folds"])
    blended_img = np.minimum(diffused_img, septa_overlayer)
    # blended_img = diffused_img
    blended_img = cv2.cvtColor(blended_img, cv2.COLOR_BGR2GRAY)
    img_path = f"{keyword}/{img_path}"
    if not os.path.exists(f"{keyword}"):
        os.mkdir(f"{keyword}")
    cv2.imwrite(img_path, blended_img)


# def process_single(f, idx_sample: tuple, vars):
#     generate_one_img(
#         idx_sample[0],
#         idx_sample[1],
#         os.path.join(
#             "fos_data/figure-4",
#             f"{idx_sample[0]:08d}.jpg",
#         ),
#         ref_path="fos_data/reference",
#         ref_poles_pool="pics/",
#         num_refs=10,  # some number
#         keyword="4"
#     ) # stand by for repackage
def generate_one_img_wrapper(wrapped):
    print(f"Running pic {wrapped['output_file']}")
    generate_one_img(
        wrapped["idx_sample"],
        wrapped["sample"],
        wrapped["output_file"],
        ref_path="fos_data/reference",
        ref_poles_pool="pics/",
        num_refs=10,  # some number
        keyword=wrapped["kwd"],  # "100k_finetune_internvl_part2"
        best_ref=wrapped["best_ref"],
    )


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
            ref_path="fos_data/reference",
            ref_poles_pool="pics/",
            num_refs=10,  # some number
            keyword=args.kwd,  # "100k_finetune_internvl_part2"
            best_ref=best_ref_list[idx_sample],
        )


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1,3,4"
    from run_gradio3_demo import crop_padding_and_resize, inference_single_image

    np.random.seed(0)
    random.seed(0)
    main()

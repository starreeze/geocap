import json
import os
import random
import sys
from io import BytesIO
from typing import Any

import cv2
import matplotlib.patches as pch
import matplotlib.pyplot as plt
import numpy as np
from iterwrap import iterate_wrapper
from PIL import Image, ImageDraw, ImageFilter

from common.args import data_args, run_args


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
        self.volution_memory = {}
        self.max_volution = max_volution
        self.center = center

    def __call__(self, shape):
        width, color, trans = self.__get_essential_info(shape)
        try:
            index = self.__special_info_validator(shape["special_info"])
        except:
            index = None
        match shape["type"]:
            case "fusiform_1":
                x_offset = shape["x_offset"]
                y_offset = shape["y_offset"]
                fc = shape["focal_length"]
                eps, ome, phi = shape["sin_params"]
                x_start = shape["x_start"]
                x_end = shape["x_end"]
                y_sim = shape["y_symmetric_axis"]

                self.__handle_fusiform_1(
                    fc,
                    x_offset,
                    y_offset,
                    eps,
                    ome,
                    phi,
                    x_start,
                    x_end,
                    y_sim,
                    shape["center"],
                    color,
                    trans,
                    width,
                )

            case "fusiform_2":
                x_offset = shape["x_offset"]
                y_offset = shape["y_offset"]
                fc = shape["focal_length"]
                eps, ome, phi = shape["sin_params"]
                power = shape["power"]
                x_start = shape["x_start"]
                x_end = shape["x_end"]

                self.__handle_fusiform_2(
                    fc,
                    x_offset,
                    y_offset,
                    power,
                    eps,
                    ome,
                    phi,
                    x_start,
                    x_end,
                    shape["center"],
                    color,
                    trans,
                    width,
                )

            case "ellipse":
                ellipse_x, ellipse_y = shape["center"]
                major = shape["major_axis"]
                minor = shape["minor_axis"]
                alpha = shape["rotation"] * 180 / np.pi

                self.__handle_ellipse(ellipse_x, ellipse_y, major, minor, alpha, width, color, trans)

            case "polygon":
                points: list = shape["points"]
                assert len(points) >= 3, "There should be more than 3 points within a polygon."

                self.__handle_polygon(points, width, color, trans)

            case "spindle":
                center_x, center_y = shape["center"]
                major = shape["major_axis"]
                minor = shape["minor_axis"]

                self.__handle_spindle(center_x, center_y, major, minor, width, color, index)

            case "curve":
                curves = shape["control_points"]

                for curve in curves:
                    self.__handle_curve(curve, width, color, index)

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
        index: int | None = None,
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
        color = (random.random(), random.random(), random.random()) if color == None else color
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
        index: int | None = None,
    ):
        color = (random.random(), random.random(), random.random()) if color == None else color
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

        x = np.concatenate((x_ru, x_mu, x_l, x_md, x_rd), axis=None)
        y = np.concatenate((y_ru, y_mu, y_l, y_md, y_rd), axis=None)

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
        color = (random.random(), random.random(), random.random()) if color == None else color

        def f(x):
            return 4 * focal_length * (x - x_offset) ** 2 + y_offset + eps * np.sin(omega * x + phi)

        x = np.linspace(x_start, x_end, 1000)
        y1 = f(x)
        y2 = 2 * y_sim - y1

        for index in range(len(x)):
            self.ax.plot((x[index], x[index]), (y1[index], y2[index]), linewidth=1, color=trans)

        self.ax.plot(x, y1, x, y2, linewidth=line_width * (self.shape[0] / 640), color=color)

        self.__keep_memory(index, x + x, y1 + y2)

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
        color = (random.random(), random.random(), random.random()) if color == None else color

        x = np.linspace(x_start, x_end, 1000)
        x_left = x[:500]
        sin_wave = eps * np.sin(omega * (x - x_start) + phi)
        y_left = (np.abs(x_left - x_offset) / (4 * focal_length)) ** (1 / power) + y_offset
        y_right = np.flip(y_left)  # 得到开口向左的上半部分
        y1 = np.concatenate([y_left, y_right]) + sin_wave
        y2 = 2 * y_offset - y1  # 得到整个纺锤形的下半部分
        for index in range(len(x)):
            self.ax.plot(
                (x[index], x[index]),
                (y1[index], y2[index]),
                linewidth=line_width * (self.shape[0] / 640),
                color=trans,
            )
        self.ax.plot(x, y1, x, y2, linewidth=line_width * (self.shape[0] / 640), color=color)

        self.__keep_memory(index, x + x, y1 + y2)

    def __handle_curve(self, control_points, width: int = 5, color=(0, 0, 0, 1), index: int | None = None):
        curve_points = []
        t_values = np.linspace(0, 1, 100)
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

        self.__keep_memory(index, curve_points[:, 0], curve_points[:, 1])

    def __get_essential_info(self, shape):
        width = self.__get_width(shape)
        color = self.__get_color(shape)
        trans = self.__get_transparency(shape)
        return width, color, trans

    def __get_width(self, shape):
        try:
            return shape["width"]
        except:
            return 5

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

    def transfer_to_cv2(self):
        buf = BytesIO()
        self.image.savefig(buf, format="png")
        buf.seek(0)

        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)

        return img

    def __special_info_validator(self, special_info):
        for i in range(10):  # How a fusi has volutions over 10 layers?
            if special_info == f"volution {i}. ":
                return i
            if special_info == "initial chamber. ":
                return -1
        print(special_info)
        assert 0  # DEBUG
        return None

    def __keep_memory(self, index, x, y):
        assert len(x) == len(y)
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
                    angle_value.append(np.arctan(y0 / x0) + (0 if y0 > 0 else np.pi))
            angle_value = np.array(angle_value)
            indices = np.argsort(angle_value)
            x = x[indices]
            y = y[indices]
            angle_value = angle_value[indices]
            self.volution_memory[index] = (x, y, angle_value)
        if index == -1:
            self.volution_memory[index] = self.center


def generate_basic_shape(shapes: list, ni: dict) -> tuple[np.ndarray, dict, int]:
    figure = Figure_Engine(max_volution=int(ni["num_volutions"]), center=ni["center"])
    for shape in shapes:
        figure(shape)
    basic_img = figure.transfer_to_cv2()
    return basic_img, figure.volution_memory, figure.max_volution


def generate_basic_mask(volution_memory: dict, filling: list) -> np.ndarray:
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
        angles = np.linspace(start_angle, end_angle, 200)
        for angle in angles:
            if angle < 0:
                angle += 2 * np.pi
            elif angle > 2 * np.pi:
                angle -= 2 * np.pi
            index_at_start = bisect(start_volution[2], angle)
            index_at_end = bisect(end_volution[2], angle)
            ax.plot(
                [start_volution[0][index_at_start], end_volution[0][index_at_end]],  # x
                [start_volution[1][index_at_start], end_volution[1][index_at_end]],  # y
                color="black",
            )
    buf = BytesIO()
    mask.savefig(buf, format="png")
    buf.seek(0)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    return img


def diffuse(img: np.ndarray, mask: np.ndarray, ref_path: str, num_refs: int) -> np.ndarray:
    from MimicBrush.run_gradio3_demo import (
        crop_padding_and_resize,
        inference_single_image,
    )

    def get_random_ref_image(ref_path: list) -> np.ndarray:
        ref_image = cv2.imread(random.choice(ref_path))
        return ref_image

    ref_paths = [os.path.join(ref_path, str(x)) for x in range(num_refs)]
    ref_image = get_random_ref_image(ref_paths)

    synthesis, depth_pred = inference_single_image(
        ref_image.copy(), img.copy(), mask.copy(), ddim_steps=60, scale=5, seed=0, enable_shape_control=True
    )

    synthesis = crop_padding_and_resize(img, synthesis)

    return synthesis.astype(np.uint8)


def generate_septa(septas: list) -> np.ndarray:
    figure = Figure_Engine()
    for septa in septas:
        figure(septa)
    return figure.transfer_to_cv2()


def generate_one_img(sample, img_path: str, ref_path: str, num_refs: int):
    basic_img, volution_memory, max_volution = generate_basic_shape(
        sample["shapes"], sample["numerical_info"]
    )
    basic_mask = generate_basic_mask(volution_memory, sample["axial_filling"])
    diffused_basic_img = diffuse(basic_img, basic_mask, ref_path, num_refs)
    poles_mask = generate_basic_mask(volution_memory, sample["poles_filling"])
    diffused_img = diffuse(diffused_basic_img, poles_mask, ref_path, num_refs)
    septa_overlayer = generate_septa(sample["septa_folds"])
    blended_img = cv2.addWeighted(diffused_img, 0.5, septa_overlayer, 0.5, 0)
    cv2.imwrite(img_path, blended_img)


def process_single(idx_sample: tuple[int, dict], f):
    generate_one_img(
        idx_sample[1],
        os.path.join(data_args.figure_dir, f"{idx_sample[0]:08d}.jpg"),
        # ref_path=data_args.reference_dir,
        ref_path="some_directory, why so serious?",
        num_refs=10,  # some number
    )


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
        assert isinstance(samples, list)
    idx_samples: list[tuple[int, dict]] = list(enumerate(samples))
    iterate_wrapper(process_single, idx_samples, num_workers=8)


if __name__ == "__main__":
    main()

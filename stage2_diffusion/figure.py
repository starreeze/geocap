import random
from io import BytesIO
from typing import Any

import cv2
import matplotlib.patches as pch
import matplotlib.pyplot as plt
import numpy as np


class Figure_Engine:
    def __init__(
        self,
        size: "tuple[float, float]" = (12.8, 12.8),
        dpi: int = 100,
        max_volution: int = 0,
        center: tuple = (0.5, 0.5),
        xkcd: bool = False,
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
        except Exception as e:
            print(e)
            index = None
        assert index is None or (-1 <= index and index <= 20)
        if shape["type"] == "fusiform_1":
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
                color,
                (0, 0, 0, 0),
                width,
                index,
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

        elif shape["type"] == "curves":
            curves = shape["control_points"]
            if isinstance(curves[0][0], float):
                curves = [curves]
            curve_buf_x = []
            curve_buf_y = []
            for curve in curves:
                x, y, center = self.__handle_curve(curve, width, color, index, trans)
                curve_buf_x.extend(x)
                curve_buf_y.extend(y)
            """
                debug_fig = plt.figure()
                ax = debug_fig.add_subplot().scatter(curve_buf_x, curve_buf_y, s=1, marker='x')
                debug_fig.savefig(f"scatter_points{index}.png")
                #"""
            self.__keep_memory(index, curve_buf_x, curve_buf_y)
            return x, y, center

    def __handle_ellipse(
        self,
        ellipse_x: float,
        ellipse_y: float,
        major: float,
        minor: float,
        alpha: float,
        line_width: int,
        color: Any,
        transparency: tuple = (0, 0, 0, 1),
        index=None,
    ):
        if major < minor:
            raise ValueError("The major axis is smaller than the minor axis, which is incorrect.")
        with plt.xkcd(randomness=20):
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

        self.__keep_memory(index, list(np.concatenate([x, x])), list(np.concatenate([y1, y2])))

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
            list(np.concatenate([x, x, x_start_lst, x_end_lst])),
            list(np.concatenate([y1, y2, y_start_lst, y_end_lst])),
        )

    def __handle_curve(
        self, control_points, width: int = 5, color=(0, 0, 0, 1), index=None, trans=(0, 0, 0, 1)
    ):
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
        with plt.xkcd(randomness=20):
            self.ax.plot(
                curve_points[:, 0], curve_points[:, 1], linewidth=width * (self.shape[0] / 640), color=color
            )
        center = np.array(
            ((curve_points[0, 0] + curve_points[-1, 0]) / 2, (curve_points[0, 1] + curve_points[-1, 1]) / 2)
        )
        assert center.shape == (2,), f"{center.shape}"
        return curve_points[:, 0], curve_points[:, 1], center

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
        except Exception:
            return 1.8

    def __get_color(self, shape):
        try:
            return shape["color"]
        except Exception:
            try:
                if shape["fill_mode"] == "border":
                    return (0, 0, 0, 1)
                else:
                    return (1, 1, 1, 1)
            except Exception:
                return (random.random(), random.random(), random.random())

    def __get_transparency(self, shape):
        try:
            if shape["fill_mode"] == "no":
                trans = (0, 0, 0, 0)
            elif shape["fill_mode"] == "white":
                trans = (1, 1, 1, 1)
            elif shape["fill_mode"] == "black":
                trans = (0, 0, 0, 1)
        except Exception:
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
        self.image.savefig(buf, format="png", transparent=True)
        # self.image.savefig(buf2,transparent=True,format="png")
        buf.seek(0)
        # buf2.seek(0)

        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
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
        assert type(x) is type(y)
        if isinstance(x, list):
            assert isinstance(y, list)
            assert len(x) == len(y)
        else:
            assert isinstance(x, float) and isinstance(y, float)
        # Suppose the center is the same
        if index is not None and index >= 0 and isinstance(x, list) and isinstance(y, list):
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

"draw geometry shapes according to generated rules"
import os, json, sys
import numpy as np
from typing import Any
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from PIL import Image, ImageDraw, ImageFilter
import random
from common.args import data_args, run_args
from common.iterwrap import iterate_wrapper

#''' Do not delete, otherwise a lot of matplotlib logs will appear.
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
#'''


class Figure:
    def __init__(
        self,
        rules: "dict",
        random_seed=None,
        randomize: bool = True,
        size: "tuple[float, float]" = (12.8, 12.8),
        dpi: int = 100,
        line_weight: int = 4,
        xkcd: bool = False,
        gradient: bool = False,
    ) -> None:
        self.rules: list = rules["shapes"]
        self.random_seed = (
            random_seed if random_seed != None else random.randint(0, 2000000)
        )
        self.randomize = randomize
        self.line_weight = line_weight
        self.image = plt.figure(figsize=size, dpi=dpi)
        self.shape = (int(size[0] * dpi), int(size[1] * dpi))
        self.size = size
        self.dpi = dpi
        self.ax = self.image.add_subplot()
        plt.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.xkcd = xkcd if randomize else False
        self.gradient = gradient

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def draw(
        self,
        color=None,
        n_redraw=None,
        n_rand_pixels=None,
        n_white_line=None,
        white_line_radius: float = 0.25,
        Gaussian_mean: float = 25,
        Gaussian_var: float = 100,
        Gaussian_proba: float = 1,
        Perlin_lattice: int = 0,
        Perlin_bias: float = 0,
        Perlin_power: float = 0,
        Perlin_proba: float = 1,
        stylish: bool = False,
        stylish_depth: int = 10,
        stylish_height: float = 3.1416 / 2.2,
        stylish_alpha: float = 3.1416 / 4,
        inline_noise: bool = True,
    ):
        for index, rule in enumerate(self.rules):
            # print(f"{index+1}/{len(self.rules)}: Handling {rule['type']}")
            if self.xkcd:
                with plt.xkcd():
                    self.__handle(rule, randomize=self.randomize, color=color)
            else:
                self.__handle(rule, randomize=self.randomize, color=color)
        # print("All rules adapted.")
        if self.randomize:
            n_white_line = (
                int(random.gauss(10, 1)) if n_white_line == None else n_white_line
            )
            self.__add_white_line(n_white_line, white_line_radius)
        self.ax.axis("off")
        # Go to PIL. PIL works better here!

        self.unprocessed_image = self.__fig2img(self.image)
        self.height, self.width = (
            self.unprocessed_image.height,
            self.unprocessed_image.width,
        )
        self.canvas = ImageDraw.Draw(self.unprocessed_image)

        # print("Monochromizing the image...")
        self.__monochromize(stylish, stylish_depth, stylish_height, stylish_alpha)

        if self.randomize:
            # print("Adding Gaussian Noise...")
            self.__add_GaussianNoise(Gaussian_mean, Gaussian_var, Gaussian_proba)
            # print("Adding Perlin Noise...")
            mask = self.__get_perlin_mask(Perlin_proba, inline_noise)
            self.__add_PerlinNoise(mask, Perlin_lattice, Perlin_power, Perlin_bias)

    def save_release(self, path: str):
        self.unprocessed_image.save(path)
        self.unprocessed_image.close()
        plt.close(self.image)

    def __fig2img(self, fig):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        img_arr = np.array(canvas.renderer.buffer_rgba())
        image = Image.fromarray(img_arr)

        return image

    def __add_GaussianNoise(self, mean: float = 0, var: float = 25, proba: float = 1):
        if random.random() < proba:
            img_array = np.array(self.unprocessed_image, dtype=float)

            noise = np.random.normal(mean, var, img_array.shape)

            processed_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            self.unprocessed_image = Image.fromarray(processed_img)


    def __get_perlin_mask(self, Perlin_proba: float = 1, inline_noise: bool = True) -> np.ndarray:
        mask = plt.figure(figsize=self.size, dpi=self.dpi)
        ax = mask.add_subplot()
        plt.subplots_adjust(0, 0, 1, 1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for idx, rule in enumerate(self.rules):
            if not random.random() < Perlin_proba:
                continue
            if rule["type"] == "ellipse":
                ellipse_x, ellipse_y = rule["center"]
                major = rule["major_axis"]
                minor = rule["minor_axis"]
                alpha = rule["rotation"] * 180 / np.pi
                ax.add_patch(
                    pch.Ellipse(
                        (ellipse_x, ellipse_y),
                        major,
                        minor,
                        angle=alpha,
                        edgecolor="black",
                        facecolor="black",
                        linewidth=self.line_weight * (self.shape[0] / 640),
                    )
                )
            if rule["type"] == "segment" and inline_noise:
                points: list = rule["points"]
                ax.plot(
                    (points[0][0], points[1][0]),
                    (points[0][1], points[1][1]),
                    color="black",
                    linewidth=self.line_weight * (self.shape[0] / 640),
                )
            if rule["type"] == "ray" and inline_noise:
                points: list = rule["points"]
                leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)


                farwise = leftwise_endpoint if points[0][0] > points[1][0] else rightwise_endpoint

                ax.plot(
                    (points[0][0], points[1][0]),
                    (farwise[0], farwise[1]),
                    color="black",
                    linewidth=self.line_weight * (self.shape[0] / 640),
                )
            if rule["type"] == "line" and inline_noise:
                points: list = rule["points"]
                leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)

                ax.plot(
                    (points[0][0], points[1][0]),
                    (leftwise_endpoint[0], leftwise_endpoint[1]),
                    color="black",
                    linewidth=self.line_weight * (self.shape[0] / 640),
                )
            if rule["type"] == "polygon":
                points: list = rule["points"]
                ax.add_patch(
                    pch.Polygon(
                        points,
                        closed=True,
                        edgecolor="black",
                        facecolor="black",
                        linewidth=self.line_weight * (self.shape[0] / 640),
                    )
                )
        img_mask = self.__fig2img(mask)
        plt.close(mask)
        mask = img_mask.convert("L")
        mask = np.array(mask)
        mask = np.where(mask > 128, 0, 1)
        return mask

    def __add_PerlinNoise(
        self, mask: np.ndarray, lattice: int = 20, power: float = 32, bias: float = 0
    ):
        def generate_perlin_noise_2d(shape, res):
            def f(t):
                return 6 * t**5 - 15 * t**4 + 10 * t**3

            delta = (res[0] / shape[0], res[1] / shape[1])
            d = (shape[0] // res[0], shape[1] // res[1])
            grid = (
                np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(
                    1, 2, 0
                )
                % 1
            )
            # Gradients
            angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
            gradients = np.dstack((np.cos(angles), np.sin(angles)))
            g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
            g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
            g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
            g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)
            # Ramps
            n00 = np.sum(grid * g00, 2)
            n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
            n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
            n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)
            # Interpolation
            t = f(grid)
            n0 = n00 * (1 - np.array(t[:, :, 0])) + t[:, :, 0] * n10
            n1 = n01 * (1 - np.array(t[:, :, 0])) + t[:, :, 0] * n11
            return np.sqrt(2) * ((1 - np.array(t[:, :, 1])) * n0 + t[:, :, 1] * n1)

        img_array = np.array(self.unprocessed_image, dtype=float)
        noise = (
            generate_perlin_noise_2d(img_array.shape, (lattice, lattice)) * power + bias
        )
        end_array = img_array + mask * noise

        processed_img = np.clip(end_array, 0, 255).astype(np.uint8)
        self.unprocessed_image = Image.fromarray(processed_img)

    def __add_white_line(self, n_line: int, max_radius: float = 0.25):
        with plt.xkcd():
            for _ in range(n_line):
                x1 = random.random()
                y1 = random.random()
                angle = random.random() * 2 * np.pi
                x2, y2 = (
                    x1 + max_radius * np.cos(angle),
                    y1 + max_radius * np.sin(angle),
                )
                self.ax.plot((x1, x2), (y1, y2), color="white", linewidth=3)

    def __handle(self, rule: "dict[str, Any]", randomize: bool, color: Any = None):
        assert (color == None) or (
            isinstance(color, list) and len(color) == 3
        ), f"Argument 'color' should be None or a 3-dimension tuple instead of {color}"
        line_width = (
            self.line_weight
            + random.randint(-self.line_weight // 2, self.line_weight // 2)
            if randomize
            else self.line_weight
        )
        if randomize:
            self.randomized_line_width = line_width
        match rule["type"]:
            case "polygon":
                points: list = rule["points"]
                assert (
                    len(points) >= 3
                ), "There should be more than 3 points within a polygon."
                if rule["fill_mode"] == "no":
                    trans = (0, 0, 0, 0)
                elif rule["fill_mode"] == "white":
                    trans = (1, 1, 1, 1)
                elif rule["fill_mode"] == "black":
                    trans = (0, 0, 0, 1)
                try:
                    line_width = rule["width"]
                except:
                    pass
                self.__handle_polygon(points, line_width, color, trans)

            case "line":
                points: list = rule["points"]
                leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)
                try:
                    line_width = rule["width"]
                except:
                    pass
                self.__handle_line(
                    (
                        (leftwise_endpoint[0], leftwise_endpoint[1]),
                        (rightwise_endpoint[0], rightwise_endpoint[1]),
                    ),
                    line_width,
                    color,
                )

            case "ray":
                points: list = rule["points"]
                leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)

                farwise = (
                    leftwise_endpoint
                    if points[0][0] > points[1][0]
                    else rightwise_endpoint
                )
                try:
                    line_width = rule["width"]
                except:
                    pass
                self.__handle_line(
                    ((points[0][0], points[0][1]), (farwise[0], farwise[1])),
                    line_width,
                    color,
                )

            case "segment":
                points: list = rule["points"]
                try:
                    line_width = rule["width"]
                except:
                    pass
                self.__handle_line(points, line_width, color)

            case "ellipse":
                ellipse_x, ellipse_y = rule["center"]
                major = rule["major_axis"]
                minor = rule["minor_axis"]
                alpha = rule["rotation"] * 180 / np.pi
                if rule["fill_mode"] == "no":
                    trans = (0, 0, 0, 0)
                elif rule["fill_mode"] == "white":
                    trans = (1, 1, 1, 1)
                elif rule["fill_mode"] == "black":
                    trans = (0, 0, 0, 1)
                try:
                    line_width = rule["width"]
                except:
                    pass
                self.__handle_ellipse(
                    ellipse_x, ellipse_y, major, minor, alpha, line_width, color, trans
                )

            case "spiral":
                # r = a + b\theta
                # x = x_0 + r\cos{\theta}
                # y = y_0 + r\sin{\theta}
                # manually draw the shape
                a: float = rule["initial_radius"]
                b: float = rule["growth_rate"]
                max_theta: float = rule["max_theta"]
                # clockwise: int = 1
                spiral_x, spiral_y = rule["center"]
                sin_params = rule["sin_params"]
                try:
                    line_width = rule["width"]
                except:
                    pass
                self.__handle_spiral(
                    spiral_x, spiral_y, a, b, max_theta, line_width, color
                )

            case "spindle":
                center_x, center_y = rule["center"]
                major = rule["major_axis"]
                minor = rule["minor_axis"]
                try:
                    line_width = rule["width"]
                except:
                    pass
                self.__handle_spindle(
                    center_x, center_y, major, minor, line_width, color
                )

            case "fusiform_1":
                x_offset = rule["x_offset"]
                y_offset = rule["y_offset"]
                fc = rule["focal_length"]
                eps, ome, phi = rule["sin_params"]
                x_start = rule["x_start"]
                x_end = rule["x_end"]
                y_sim = rule["y_symmetric_axis"]

                if rule["fill_mode"] == "no":
                    trans = (0, 0, 0, 0)
                elif rule["fill_mode"] == "white":
                    trans = (1, 1, 1, 1)
                elif rule["fill_mode"] == "black":
                    trans = (0, 0, 0, 1)
                try:
                    line_width = rule["width"]
                except:
                    pass
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
                    trans,
                    line_width,
                )

            case "fusiform_2":
                x_offset = rule["x_offset"]
                y_offset = rule["y_offset"]
                fc = rule["focal_length"]
                eps, ome, phi = rule["sin_params"]
                power = rule["power"]
                x_start = rule["x_start"]
                x_end = rule["x_end"]

                if rule["fill_mode"] == "no":
                    trans = (0, 0, 0, 0)
                elif rule["fill_mode"] == "white":
                    trans = (1, 1, 1, 1)
                elif rule["fill_mode"] == "black":
                    trans = (0, 0, 0, 1)

                try:
                    line_width = rule["width"]
                except:
                    pass

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
                    color,
                    trans,
                    line_width,
                )

            case "curves":
                curves = rule["control_points"]
                try:
                    line_width = rule["width"]
                except:
                    pass  # keep the original data
                for curve in curves:
                    self.__handle_curve(curve, line_width)

            case _:
                raise ValueError(f"{rule['type']} is not any valid rule.")

    # It's very likely that 'patch' should be truncated and use plot instead to complete the change of width

    def __handle_line(self, points, line_width: int, color: Any):
        color = (
            (
                random.random(),
                random.random(),
                random.random(),
            )
            if color == None
            else color
        )
        if self.xkcd:
            self.ax.plot(
                (points[0][0], points[1][0]),
                (points[0][1], points[1][1]),
                linewidth=line_width * (self.shape[0] / 640),
                color=color,
            )
        elif self.gradient:
            ln_wths = np.linspace(line_width / 2, line_width + line_width / 2, 50)
            x = np.linspace(points[0][0], points[1][0], 50)
            y = np.linspace(points[0][1], points[1][1], 50)
            for i in range(50):
                self.ax.plot(
                    x[i : i + 2],
                    y[i : i + 2],
                    linewidth=ln_wths[i] * (self.shape[0] / 640),
                    color=np.clip(color + np.array((0.005, 0.005, 0.005)) * i, 0, 1),
                )
        else:
            self.ax.plot(
                (points[0][0], points[1][0]),
                (points[0][1], points[1][1]),
                linewidth=line_width * (self.shape[0] / 640),
                color=color,
            )

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
    ):
        color = (
            (random.random(), random.random(), random.random())
            if color == None
            else color
        )
        if major < minor:
            raise ValueError(
                "The major axis is smaller than the minor axis, which is incorrect."
            )

        """
        major = major / 2
        minor = minor / 2

        theta = np.linspace(0, np.pi * 2, 720)
        radius = np.sqrt(
            1
            / (
                ((np.cos(theta + alpha / 360 * 2 * np.pi)) ** 2) / major**2
                + ((np.sin(theta + alpha / 360 * 2 * np.pi)) ** 2) / minor**2
            )
        )
        x = radius * np.cos(theta) + ellipse_x
        y = radius * np.sin(theta) + ellipse_y
        self.ax.plot(x, y, color=color)

        major *= 2
        minor *= 2
        """
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

    def __handle_polygon(
        self, points: list, line_width: int, color: Any, trans: tuple = (0, 0, 0, 0)
    ):
        color = (
            (
                random.random(),
                random.random(),
                random.random(),
            )
            if color == None
            else color
        )
        self.ax.add_patch(
            pch.Polygon(
                points,
                closed=True,
                edgecolor=color,
                linewidth=line_width * (self.shape[0] / 640),
                facecolor=trans,
            )
        )

    def __handle_spiral(
        self,
        spiral_x: float,
        spiral_y: float,
        a: float,
        b: float,
        max_theta: float,
        line_width: int,
        color: Any,
    ):
        color = (
            (
                random.random(),
                random.random(),
                random.random(),
            )
            if color == None
            else color
        )
        theta = np.arange(0, max_theta, 0.01)
        x = (a + b * theta) * np.cos(theta)
        y = (a + b * theta) * np.sin(theta)
        x += spiral_x
        y += spiral_y
        self.ax.plot(x, y, color=color, linewidth=line_width * (self.shape[0] / 640))

    def __handle_spindle(
        self,
        center_x: float,
        center_y: float,
        major_axis: float,
        minor_axis: float,
        line_width: int,
        color: Any,
    ):
        color = (
            (
                random.random(),
                random.random(),
                random.random(),
            )
            if color == None
            else color
        )
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
    ):
        color = (
            (
                random.random(),
                random.random(),
                random.random(),
            )
            if color == None
            else color
        )

        def f(x):
            return (
                4 * focal_length * (x - x_offset) ** 2
                + y_offset
                + eps * np.sin(omega * x + phi)
            )

        x = np.linspace(x_start, x_end, 1000)
        y1 = f(x)
        y2 = 2 * y_sim - y1
        """
        for index in range(len(x)):
            self.ax.plot(
                (x[index], x[index]), (y1[index], y2[index]), linewidth=1, color=trans
            )
        """
        self.ax.plot(
            x, y1, x, y2, linewidth=line_width * (self.shape[0] / 640), color=color
        )

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
    ):
        color = (
            (
                random.random(),
                random.random(),
                random.random(),
            )
            if color == None
            else color
        )

        x = np.linspace(x_start, x_end, 1000)
        x_left = x[:500]
        sin_wave = eps * np.sin(omega * (x - x_start) + phi)
        y_left = (np.abs(x_left - x_offset) / (4 * focal_length)) ** (
            1 / power
        ) + y_offset
        y_right = np.flip(y_left)  # 得到开口向左的上半部分
        y1 = np.concatenate([y_left, y_right]) + sin_wave
        y2 = 2 * y_offset - y1  # 得到整个纺锤形的下半部分
        # """
        if self.xkcd:
            plt.xkcd()
        for index in range(len(x)):
            self.ax.plot(
                (x[index], x[index]),
                (y1[index], y2[index]),
                linewidth=line_width * (self.shape[0] / 640),
                color=trans,
            )
        if self.xkcd:
            plt.xkcd()
        # """
        self.ax.plot(
            x, y1, x, y2, linewidth=line_width * (self.shape[0] / 640), color=color
        )

    def __handle_curve(self, control_points, width: int = 5):
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
            curve_points[:, 0],
            curve_points[:, 1],
            linewidth=width * (self.shape[0] / 640),
        )

    def __line_extend(self, points: list) -> tuple:
        if points[0][0] == points[1][0]:
            return ((points[0][0], 0), (points[1][0], 1))
        line_k = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
        line_b = points[0][1] - line_k * points[0][0]

        if (line_b >= 0) and (line_b <= 1):
            leftwise_endpoint = (0, line_b)
        elif line_b < 0:
            leftwise_endpoint = (-line_b / line_k, 0)
        else:  # line_b > self.height:
            leftwise_endpoint = (
                (1 - line_b) / line_k,
                1,
            )

        if line_k * 1 + line_b >= 0 and line_k * 1 + line_b <= 1:
            rightwise_endpoint = (1, line_k * 1 + line_b)
        elif line_k * 1 + line_b < 0:
            rightwise_endpoint = (-line_b / line_k, 0)
        else:  # line_k * self.width + line_b > self.height:
            rightwise_endpoint = (
                (1 - line_b) / line_k,
                1,
            )

        return (leftwise_endpoint, rightwise_endpoint)

    def __monochromize(
        self,
        stylish: bool = False,
        depth: int = 10,
        height: float = 3.1416 / 2.2,
        alpha: float = 3.1416 / 4,
    ):
        self.unprocessed_image = self.unprocessed_image.convert("L")
        if not stylish:
            return
        data = np.array(self.unprocessed_image)
        grad_x, grad_y = np.gradient(data)

        grad_x /= 100 / depth
        grad_y /= 100 / depth

        dx = np.cos(height) * np.cos(alpha)
        dy = np.cos(height) * np.sin(alpha)
        dz = np.sin(height)

        remap = np.sqrt(grad_x**2 + grad_y**2 + 1)
        uni_x = grad_x / remap
        uni_y = grad_y / remap
        uni_z = 1 / remap

        self.unprocessed_image = Image.fromarray(
            (255 * (dx * uni_x + dy * uni_y + dz * uni_z)).clip(0, 255).astype("uint8")
        )

    def close(self):
        plt.close(self.image)


def draw_figure(rules: "dict", path: str):
    # TODO apply rules to draw shapes (DONE)
    # TODO control their line weight and curves (MANUALLY)

    # TODO add various backgrounds and noise (HOW?)
    figure = Figure(rules, random_seed=0, xkcd=True)
    figure.draw(stylish=True)
    figure.save_release(path)
    figure.close()


def process_single(f, idx_sample: tuple[int, dict], vars):
    draw_figure(
        idx_sample[1], os.path.join(data_args.figure_dir, f"{idx_sample[0]:08d}.jpg")
    )


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
        assert isinstance(samples, list)
    # for idx_sample, sample in enumerate(samples):
    #     draw_figure(sample, os.path.join(data_args.figure_dir, f"{idx_sample:08d}.jpg"))
    iterate_wrapper(process_single, list(enumerate(samples)), num_workers=8)


if __name__ == "__main__":
    main()

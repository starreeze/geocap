"draw geometry shapes according to generated rules"
import os, json, sys
import numpy as np
from typing import Any

# Debug
#'''
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
#'''
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from PIL import Image, ImageDraw, ImageFilter
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import data_args


class Figure:
    def __init__(
        self,
        rules: "list[dict[str, Any]]",
        random_seed: int,
        randomize: bool = True,
        size: "tuple[float, float]" = (6.4, 6.4),
        dpi: int = 100,
        line_weight: int = 4,
        xkcd: bool = False,
    ) -> None:
        self.rules = rules
        self.random_seed = random_seed
        self.randomize = randomize
        self.line_weight = line_weight
        self.image = plt.figure(figsize=size, dpi=dpi)
        self.ax = self.image.add_subplot()
        plt.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.xkcd = xkcd
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def draw(
        self,
        color=None,
        n_rand_pixels=None,
        n_white_line=None,
        Gaussian_mean=30,
        Gaussian_var=300,
        stylish: bool = False,
    ):
        for index, rule in enumerate(self.rules):
            print(f"{index+1}/{len(self.rules)}: Handling {rule['type']}")
            self.__handle(rule, randomize=self.randomize, color=color)
        print("All rules adapted.")
        n_white_line = (
            int(random.gauss(10, 1)) if n_white_line == None else n_white_line
        )
        self.__add_white_line(n_white_line)
        self.ax.axis("off")
        # Go to PIL. PIL works better here!

        self.unprocessed_image = self.__fig2img()
        self.canvas = ImageDraw.Draw(self.unprocessed_image)

        if self.randomize:
            print("Adding Noise...")
            self.__add_noise(n_rand_pixels)
        print("Monochromizing the image...")
        self.__monochromize(stylish)
        self.__add_GaussianNoise(Gaussian_mean, Gaussian_var)

    def save(self, path: str):
        self.unprocessed_image.save(path)

    def __fig2img(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        canvas = FigureCanvasAgg(self.image)
        canvas.draw()

        img_arr = np.array(canvas.renderer.buffer_rgba())
        image = Image.fromarray(img_arr)

        self.width, self.height = image.width, image.height

        return image

    def __add_noise(
        self,
        n_rand_pixels=None,
    ):
        assert (
            self.randomize
        ), "Function 'add_noise' is disabled whilst randomize==False"
        n_rand_pixels = (
            int(random.gauss(100, 5)) if n_rand_pixels == None else n_rand_pixels
        )
        self.__add_random_pixels(n_pixels=n_rand_pixels)

    def __add_random_pixels(self, n_pixels: int):
        for _ in range(n_pixels):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            self.canvas.point((x, y), fill="black")

    def __add_GaussianNoise(self, mean: float = 0, var: float = 25):
        img_array = np.array(self.unprocessed_image, dtype=float)
        noise = np.random.normal(mean, var**0.5, img_array.shape)
        processed_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        self.unprocessed_image = Image.fromarray(processed_img)
        self.unprocessed_image = self.unprocessed_image.filter(
            ImageFilter.GaussianBlur(0.5)
        )

    def __add_white_line(self, n_line: int):
        with plt.xkcd():
            for _ in range(n_line):
                x1 = random.random()
                y1 = random.random()
                x2 = random.random()
                y2 = random.random()
                self.ax.plot((x1, x2), (y1, y2), color="white", linewidth=3)

    def __handle(self, rule: "dict[str, Any]", randomize: bool, color: Any = None):
        assert (color == None) or (
            isinstance(color, tuple) and len(color) == 3
        ), "Argument 'color' should be None or a 3-dimension tuple."
        line_width = (
            self.line_weight
            + random.randint(-self.line_weight // 2, self.line_weight // 2)
            if randomize
            else self.line_weight
        )
        if randomize:
            self.randomized_line_width = line_width
        if self.xkcd:
            plt.xkcd()
        match rule["type"]:
            case "polygon":
                points: list = rule["points"]
                assert (
                    len(points) >= 3
                ), "There should be more than 3 points within a polygon."
                self.__handle_polygon(points, line_width, color)

            case "line":
                points: list = rule["points"]
                leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)
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

                self.__handle_line(
                    ((points[0][0], points[0][1]), (farwise[0], farwise[1])),
                    line_width,
                    color,
                )
            case "segment":
                points: list = rule["points"]
                self.__handle_line(points, line_width, color)

            case "ellipse":
                ellipse_x, ellipse_y = rule["center"]
                major = rule["major_axis"]
                minor = rule["minor_axis"]
                alpha = rule["rotation"] * 180 / np.pi
                self.__handle_ellipse(
                    ellipse_x, ellipse_y, major, minor, alpha, line_width, color
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
                self.__handle_spiral(
                    spiral_x, spiral_y, a, b, max_theta, line_width, color
                )

            case _:
                raise ValueError(f"{rule['type']} is not any valid rule.")

    # It's very likely that 'patch' should be truncated and use plot instead to complete the change of width

    def __handle_line(self, points, line_width: int, color: Any):
        if self.xkcd:
            self.ax.plot(
                (points[0][0], points[1][0]),
                (points[0][1], points[1][1]),
                linewidth=line_width,
                color=color,
            )
        else:
            ln_wths = np.linspace(line_width / 2, line_width + line_width / 2, 50)
            x = np.linspace(points[0][0], points[1][0], 50)
            y = np.linspace(points[0][1], points[1][1], 50)
            color = (
                (
                    random.random(),
                    random.random(),
                    random.random(),
                )
                if color == None
                else color
            )
            for i in range(50):
                self.ax.plot(
                    x[i : i + 2],
                    y[i : i + 2],
                    linewidth=ln_wths[i],
                    color=color,
                )

    def __handle_ellipse(
        self,
        ellipse_x: float,
        ellipse_y: float,
        major: int,
        minor: int,
        alpha: float,
        line_width: int,
        color: Any,
    ):
        self.ax.add_patch(
            pch.Ellipse(
                (ellipse_x, ellipse_y),
                major,
                minor,
                angle=alpha,
                edgecolor=(
                    (random.random(), random.random(), random.random())
                    if color == None
                    else color
                ),
                facecolor=(0, 0, 0, 0),
                linewidth=line_width,
            )
        )

    def __handle_polygon(self, points: list, line_width: int, color: Any):
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
                linewidth=line_width,
                facecolor=(0, 0, 0, 0),
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
        self.ax.plot(x, y, color=color, linewidth=line_width)

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


def draw_figure(rules: "list[dict[str, Any]]", path: str):
    # TODO apply rules to draw shapes (DONE)
    # TODO control their line weight and curves (MANUALLY)

    # TODO add various backgrounds and noise (HOW?)
    figure = Figure(rules, random_seed=0, xkcd=True)
    figure.draw(n_rand_pixels=80)
    figure.save(path)


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
    for i, sample in enumerate(samples):
        draw_figure(sample, f"dataset/pictures/{i}_PLT.png")


if __name__ == "__main__":
    main()

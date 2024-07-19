"draw geometry shapes according to generated rules"
import os, json, sys
import math
import numpy as np
from typing import Any
from PIL import Image, ImageDraw
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
        size: "tuple[int, int]" = (1024, 1024),
        background: "tuple[int, int, int]" = (255, 255, 255),
        line_weight: int = 4,
    ) -> None:
        self.rules = rules
        self.random_seed = random_seed
        self.randomize = randomize
        self.width = size[0]
        self.height = size[1]
        self.background = background
        self.line_weight = line_weight
        self.image = Image.new(mode="RGB", size=(self.width, self.height), color=self.background)
        self.canvas = ImageDraw.Draw(self.image, mode="RGB")
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def draw(
        self,
        color=None,
        n_redraw=None,
        n_rand_pixels=None,
        n_white_line=None,
        Gaussian_mean=25,
        Gaussian_var=100,
    ):
        for index, rule in enumerate(self.rules):
            print(f"{index+1}/{len(self.rules)}: Handling {rule['type']}")
            self.__handle(rule, randomize=self.randomize, color=color)
        print("All rules adapted.")
        if self.randomize:
            print("Adding Noise...")
            self.__add_noise(n_redraw, n_rand_pixels, n_white_line, Gaussian_mean, Gaussian_var)
        print("Monochromizing the image...")
        self.__monochromize()

    def save(self, path: str):
        self.image.save(fp=path)

    def __add_noise(
        self,
        n_redraw=None,
        n_rand_pixels=None,
        n_white_line=None,
        Gaussian_mean=25,
        Gaussian_var=100,
    ):
        assert self.randomize, "Function 'add_noise' is disabled whilst randomize==False"
        try:
            rdm_lw = self.randomized_line_width
        except:
            raise AttributeError("Must firstly run 'draw' to create attribute 'randomized_line_width'")
        n_redraw = int(random.gauss(len(self.rules) // 2, len(self.rules) // 20)) if n_redraw == None else n_redraw
        n_rand_pixels = int(random.gauss(100, 5)) if n_rand_pixels == None else n_rand_pixels
        n_white_line = int(random.gauss(10, 1)) if n_white_line == None else n_white_line
        for index, rule in enumerate(random.sample(self.rules, n_redraw)):
            print(f"Redrawing #{index}: {rule['type']}")
            self.__redraw(rule)

        self.__add_random_pixels(n_pixels=n_rand_pixels)
        self.__add_GaussianNoise(Gaussian_mean, Gaussian_var)
        self.__add_white_line(n_white_line)

    def __redraw(self, rule: "dict[str, Any]"):
        self.line_weight = self.randomized_line_width + 2
        # TODO: redrawing type "line" needs to adjust its width in a certain range
        self.__handle(rule, randomize=False, color=None)

    def __add_random_pixels(self, n_pixels: int):
        for _ in range(n_pixels):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            self.canvas.point((x, y), fill="black")

    def __add_GaussianNoise(self, mean: float = 0, var: float = 25):
        img_array = np.array(self.image, dtype=float)
        noise = np.random.normal(mean, var**0.5, img_array.shape)
        processed_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        self.image = Image.fromarray(processed_img)

    def __add_white_line(self, n_line: int):
        for _ in range(n_line):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            self.canvas.line((x1, y1, x2, y2), fill="white", width=5)

    def __handle(self, rule: "dict[str, Any]", randomize: bool, color: Any):
        assert (color == None) or (
            isinstance(color, tuple) and len(color) == 3
        ), "Argument 'color' should be set empty, or a 3-dimension tuple."
        line_width = (
            self.line_weight + random.randint(-self.line_weight // 2, self.line_weight // 2)
            if randomize
            else self.line_weight
        )
        if randomize:
            self.randomized_line_width = line_width
        match rule["type"]:
            case "polygon":
                points: list = [self.__translate(point) for point in rule["points"]]
                assert len(points) >= 3, "There should be more than 3 points within a polygon."
                self.__handle_polygon(points, line_width, color)

            case "line":
                points: list = [self.__translate(point) for point in rule["points"]]
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
                points: list = [self.__translate(point) for point in rule["points"]]
                leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)

                farwise = leftwise_endpoint if points[0][0] > points[1][0] else rightwise_endpoint

                self.__handle_line(
                    ((points[0][0], points[0][1]), (farwise[0], farwise[1])),
                    line_width,
                    color,
                )
            case "segment":
                points: list = [self.__translate(point) for point in rule["points"]]
                self.__handle_line(points, line_width, color)

            case "ellipse":
                ellipse_x, ellipse_y = self.__translate(rule["center"])
                major = int(rule["major_axis"] * self.width)
                minor = int(rule["minor_axis"] * self.height)
                alpha = rule["rotation"]
                self.__handle_ellipse(ellipse_x, ellipse_y, major, minor, alpha, line_width, color)

            case "spiral":
                # r = a + b\theta
                # x = x_0 + r\cos{\theta}
                # y = y_0 + r\sin{\theta}
                # manually draw the shape
                # A bit buggy when a \approx 0 and b is relatively big, about 0.1
                a: float = rule["initial_radius"] * self.width
                b: float = rule["growth_rate"] * self.width
                max_theta: float = rule["max_theta"]
                # clockwise: int = 1
                spiral_x, spiral_y = self.__translate(rule["center"])
                # self.canvas.point((spiral_x, spiral_y), fill="red")
                self.__handle_spiral(spiral_x, spiral_y, a, b, max_theta, line_width, color)

            case _:
                raise ValueError(f"{rule['type']} is not any valid rule.")

    def __handle_line(self, points, line_width: int, color: Any):
        self.canvas.line(
            xy=(points[0][0], points[0][1], points[1][0], points[1][1]),
            width=line_width,
            fill=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255) if color == None else color,
            ),
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
        ellipse_x -= major // 2
        ellipse_y -= major // 2

        im_ellipse_bkgrnd = (
            self.background[0],
            self.background[1],
            self.background[2],
            0,
        )
        im_ellipse = Image.new("RGBA", (major, major), im_ellipse_bkgrnd)
        cvs_ellipse = ImageDraw.Draw(im_ellipse, "RGBA")

        cvs_ellipse.ellipse(
            (0, major // 2 - minor // 2, major, major // 2 + minor // 2),
            fill=self.background,
            outline=(
                (
                    random.randint(0, 255),
                    random.randint(0, 255),
                    random.randint(0, 255),
                    255,
                )
                if color == None
                else list(color).append(255)
            ),
            width=line_width,
        )  # original ellipse

        rotated = im_ellipse.rotate(alpha * 180 / 3.1416)  # precision should be enough
        rx, ry = rotated.size

        # rotated.save("debugging_rotated.png")

        self.image.paste(
            rotated,
            (
                int(ellipse_x),
                int(ellipse_y),
                int(ellipse_x + rx),
                int(ellipse_y + ry),
            ),
            mask=rotated,
        )

    def __handle_polygon(self, points: list, line_width: int, color: Any):
        for index in range(len(points)):
            self.canvas.line(
                xy=(
                    points[index][0],
                    points[index][1],
                    points[(index + 1) % len(points)][0],
                    points[(index + 1) % len(points)][1],
                ),
                width=(line_width),
                fill=(
                    (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    if color == None
                    else color
                ),
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
        theta: float = 0

        if a <= 0.01:
            a = 1  # promise r \neq 0

        while theta <= max_theta:
            c_theta = math.cos(theta)
            s_theta = math.sin(theta)
            r = a + b * theta

            for w in range(line_width):
                x = spiral_x + (r - line_width / 2 + w) * c_theta
                y = spiral_y - (r - line_width / 2 + w) * s_theta
                self.canvas.point(
                    (x, y),
                    fill=(
                        (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        )
                        if color == None
                        else color
                    ),
                )
            theta += math.atan(1 / r) if r < 10 else 1 / r  # arctan(1/r) \approx 1/r, speed up

    def __line_extend(self, points: list) -> tuple:
        if points[0][0] == points[1][0]:
            return ((points[0][0], 0), (points[1][0], self.height))
        line_k = (points[0][1] - points[1][1]) / (points[0][0] - points[1][0])
        line_b = points[0][1] - line_k * points[0][0]

        if (line_b >= 0) and (line_b <= self.height):
            leftwise_endpoint = (0, line_b)
        elif line_b < 0:
            leftwise_endpoint = (-line_b / line_k, 0)
        else:  # line_b > self.height:
            leftwise_endpoint = (
                (self.height - line_b) / line_k,
                self.height,
            )

        if line_k * self.width + line_b >= 0 and line_k * self.width + line_b <= self.height:
            rightwise_endpoint = (self.width, line_k * self.width + line_b)
        elif line_k * self.width + line_b < 0:
            rightwise_endpoint = (-line_b / line_k, 0)
        else:  # line_k * self.width + line_b > self.height:
            rightwise_endpoint = (
                (self.height - line_b) / line_k,
                self.height,
            )

        return (leftwise_endpoint, rightwise_endpoint)

    def __translate(self, point: tuple) -> tuple:
        return (self.width * point[0], self.height * point[1])

    def __monochromize(self):
        self.image = self.image.convert("L")


def draw_figure(rules: "list[dict[str, Any]]", path: str):
    # TODO apply rules to draw shapes (DONE)
    # TODO control their line weight and curves (MANUALLY)

    # TODO add various backgrounds and noise (HOW?)
    figure = Figure(rules, random_seed=0)
    figure.draw()
    figure.save(path)


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
    for i, sample in enumerate(samples):
        draw_figure(sample, f"dataset/pictures/{i}.png")


if __name__ == "__main__":
    main()

"draw geometry shapes according to generated rules"
import os, json, sys
import numpy as np
from typing import Any
from PIL import Image, ImageDraw
import random
from common.args import data_args


class Figure:
    def __init__(
        self,
        rules: "dict",
        random_seed=None,
        randomize: bool = True,
        size: "tuple[int, int]" = (1024, 1024),
        background: "tuple[int, int, int]" = (255, 255, 255),
        line_weight: int = 4,
    ) -> None:
        self.rules: list = rules["shapes"]
        self.random_seed = random_seed
        self.randomize = randomize
        self.width = size[0]
        self.height = size[1]
        self.background = background
        self.line_weight = line_weight
        self.image = Image.new(
            mode="RGB", size=(self.width, self.height), color=self.background
        )
        self.canvas = ImageDraw.Draw(self.image, mode="RGB")
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def draw(
        self,
        color=None,
        n_redraw=None,
        n_rand_pixels=None,
        n_white_line=None,
        Gaussian_mean: float = 25,
        Gaussian_var: float = 100,
        Perlin_lattice: int = 0,
        Perlin_bias: float = 0,
        Perlin_power: float = 0,
        stylish: bool = False,
    ):
        for index, rule in enumerate(self.rules):
            print(f"{index+1}/{len(self.rules)}: Handling {rule['type']}")
            self.__handle(rule, randomize=self.randomize, color=color)
        print("All rules adapted.")
        if self.randomize:
            print("Adding Noise...")
            self.__add_noise(
                n_redraw, n_rand_pixels, n_white_line, Gaussian_mean, Gaussian_var
            )
        print("Monochromizing the image...")
        self.__monochromize(stylish)

    def save(self, path: str):
        self.image.save(fp=path)

    def __add_noise(
        self,
        n_redraw=None,
        n_rand_pixels=None,
        n_white_line=None,
        Gaussian_mean: float = 25,
        Gaussian_var: float = 100,
    ):
        assert (
            self.randomize
        ), "Function 'add_noise' is disabled whilst randomize==False"
        try:
            rdm_lw = self.randomized_line_width
        except:
            raise AttributeError(
                "Must firstly run 'draw' to create attribute 'randomized_line_width'"
            )
        n_redraw = (
            int(random.gauss(len(self.rules) // 2, len(self.rules) // 20))
            if n_redraw == None
            else n_redraw
        )
        n_rand_pixels = (
            int(random.gauss(100, 5)) if n_rand_pixels == None else n_rand_pixels
        )
        n_white_line = (
            int(random.gauss(10, 1)) if n_white_line == None else n_white_line
        )
        self.__redraw(n_redraw)
        self.__add_random_pixels(n_pixels=n_rand_pixels)
        self.__add_white_line(n_white_line)
        self.__add_GaussianNoise(Gaussian_mean, Gaussian_var)

    def __redraw(self, n_redraw: int):
        n_redraw = n_redraw if n_redraw < len(self.rules) else len(self.rules)
        for index, rule in enumerate(random.sample(self.rules, n_redraw)):
            print(f"Redrawing #{index}: {rule['type']}")
            match rule["type"]:
                case "segment":
                    points = [self.__translate(i) for i in rule["points"]]
                    points = [
                        (
                            point[0],
                            point[1],
                            int(self.randomized_line_width + random.gauss(10, 10)),
                        )
                        for point in points
                    ]
                    self.__redraw_line(mode="2_points_control", control_points=points)
                case "line":
                    points: list = [self.__translate(point) for point in rule["points"]]
                    leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)
                    self.__redraw_line(
                        control_points=[
                            (
                                leftwise_endpoint[0],
                                leftwise_endpoint[1],
                                int(self.randomized_line_width + random.gauss(10, 10)),
                            ),
                            (
                                rightwise_endpoint[0],
                                rightwise_endpoint[1],
                                int(self.randomized_line_width + random.gauss(10, 10)),
                            ),
                        ],
                        mode="2_points_control",
                    )
                case "ray":
                    points: list = [self.__translate(point) for point in rule["points"]]
                    leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)

                    farwise = (
                        leftwise_endpoint
                        if points[0][0] > points[1][0]
                        else rightwise_endpoint
                    )

                    self.__redraw_line(
                        control_points=[
                            (
                                points[0][0],
                                points[0][1],
                                int(self.randomized_line_width + random.gauss(10, 10)),
                            ),
                            (
                                farwise[0],
                                farwise[1],
                                int(self.randomized_line_width + random.gauss(10, 10)),
                            ),
                        ],
                        mode="2_points_control",
                    )
                case "polygon":
                    points = [self.__translate(i) for i in rule["points"]]
                    points = [
                        (
                            point[0],
                            point[1],
                            int(self.randomized_line_width + random.gauss(10, 10)),
                        )
                        for point in points
                    ]
                    self.__redraw_polygon(points)
                case _:
                    self.line_weight = int(
                        self.randomized_line_width + random.gauss(20, 10)
                    )
                    self.__handle(rule, randomize=False)

    def __redraw_line(
        self,
        points=[],
        mode: str = "auto",
        control_points: "list[tuple[float, float, int]]" = [],
        ascensions: "list[tuple[float, float, int]]" = [],
        color=None,
    ):
        match mode:
            case "auto":
                self.canvas.line(
                    (points[0][0], points[0][1], points[1][0], points[1][1]),
                    fill=(
                        (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        )
                        if color == None
                        else color
                    ),
                    width=int(self.randomized_line_width + random.gauss(20, 10)),
                )
            case "2_points_control":
                # Override the points argument
                assert (
                    len(control_points) == 2
                ), "You must give exactly two points' info in the argument"
                width = [
                    control_points[i][2] if control_points[i][2] > 0 else 1
                    for i in range(2)
                ]
                x = [control_points[i][0] for i in range(2)]
                y = [control_points[i][1] for i in range(2)]
                if width[0] == width[1]:
                    self.canvas.line(
                        xy=(x[0], y[0], x[1], y[1]),
                        fill=(
                            (
                                random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255),
                            )
                            if color == None
                            else color
                        ),
                        width=width[1],
                    )
                    return
                length_of_line = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
                pixels_per_ascend = length_of_line // np.absolute(width[0] - width[1])
                if pixels_per_ascend == 0:
                    print(
                        "The line is too short to perform this redraw attempt. Reperform the attempt in auto mode."
                    )
                    self.__redraw_line(points, mode="auto")
                    return
                begin_point = 0 if width[0] < width[1] else 1
                end_point = 1 if width[0] < width[1] else 0
                uni_vec = (
                    (x[end_point] - x[begin_point]) / length_of_line,
                    (y[end_point] - y[begin_point]) / length_of_line,
                )
                n_ascend = width[end_point] - width[begin_point]
                color = (
                    (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    if color == None
                    else color
                )
                for i in range(n_ascend - 1):
                    xy = (
                        x[begin_point] + i * pixels_per_ascend * uni_vec[0],
                        y[begin_point] + i * pixels_per_ascend * uni_vec[1],
                        x[begin_point] + (i + 1) * pixels_per_ascend * uni_vec[0],
                        y[begin_point] + (i + 1) * pixels_per_ascend * uni_vec[1],
                    )
                    self.canvas.line(
                        xy=xy,
                        fill=color,
                        width=width[begin_point] + i,
                    )
                self.canvas.line(
                    xy=(
                        x[begin_point]
                        + (n_ascend - 1) * pixels_per_ascend * uni_vec[0],
                        y[begin_point]
                        + (n_ascend - 1) * pixels_per_ascend * uni_vec[1],
                        x[end_point],
                        y[end_point],
                    ),
                    fill=color,
                    width=width[end_point],
                )

            case "manual":
                color = (
                    (
                        random.randint(0, 255),
                        random.randint(0, 255),
                        random.randint(0, 255),
                    )
                    if color == None
                    else color
                )
                for i in range(len(ascensions) - 1):
                    self.canvas.line(
                        (
                            ascensions[i][0],
                            ascensions[i][1],
                            ascensions[i + 1][0],
                            ascensions[i + 1][1],
                        ),
                        width=ascensions[i][2],
                        fill=color,
                    )
            case _:
                raise ValueError(
                    f"Invalid argument mode = {mode}, must be 'auto', '2_points_control', or 'manual'."
                )

    def __redraw_polygon(self, points):
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for index in range(len(points)):
            self.__redraw_line(
                points=None,
                control_points=[
                    (points[index][0], points[index][1], points[index][2]),
                    (
                        points[(index + 1) % len(points)][0],
                        points[(index + 1) % len(points)][1],
                        points[(index + 1) % len(points)][2],
                    ),
                ],
                mode="2_points_control",
                color=color,
            )
            """
            self.canvas.circle(
                xy=(points[index][0], points[index][1]),
                radius=points[index][2] / 2 if points[index][2] > 0 else 1,
                fill=color,
                outline=color,
            )
            """

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
            self.canvas.line(
                (x1, y1, x2, y2), fill="white", width=self.randomized_line_width
            )

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
        match rule["type"]:
            case "polygon":
                points: list = [self.__translate(point) for point in rule["points"]]
                assert (
                    len(points) >= 3
                ), "There should be more than 3 points within a polygon."
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
                points: list = [self.__translate(point) for point in rule["points"]]
                self.__handle_line(points, line_width, color)

            case "ellipse":
                ellipse_x, ellipse_y = self.__translate(rule["center"])
                major = int(rule["major_axis"] * self.width)
                minor = int(rule["minor_axis"] * self.height)
                alpha = rule["rotation"]
                self.__handle_ellipse(
                    ellipse_x, ellipse_y, major, minor, alpha, line_width, color
                )

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
                self.__handle_spiral(
                    spiral_x, spiral_y, a, b, max_theta, line_width, color
                )

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
            fill=(self.background[0], self.background[1], self.background[2], 0),
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

        rotated = im_ellipse.rotate(-alpha * 180 / 3.1416)  # precision should be enough
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
        color = (
            (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            if color == None
            else color
        )
        for index in range(len(points)):
            self.canvas.line(
                xy=(
                    points[index][0],
                    points[index][1],
                    points[(index + 1) % len(points)][0],
                    points[(index + 1) % len(points)][1],
                ),
                width=(line_width),
                fill=(color),
            )
            # self.canvas.circle(xy=points[index], radius=line_width / 2, fill=color, outline=color)

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

        color = (
            (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            )
            if color == None
            else color
        )

        if a <= 0.01:
            a = 1  # promise r \neq 0

        while theta <= max_theta:
            c_theta = np.cos(theta)
            s_theta = np.sin(theta)
            r = a + b * theta

            for w in range(line_width):
                x = spiral_x + (r - line_width / 2 + w) * c_theta
                y = spiral_y - (r - line_width / 2 + w) * s_theta
                self.canvas.point(
                    (x, y),
                    fill=(color),
                )
            theta += (
                np.arctan(1 / r) if r < 10 else 1 / r
            )  # arctan(1/r) \approx 1/r, speed up

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

        if (
            line_k * self.width + line_b >= 0
            and line_k * self.width + line_b <= self.height
        ):
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

    def __monochromize(
        self,
        stylish: bool = False,
        depth: int = 10,
        height: float = 3.1416 / 2.2,
        alpha: float = 3.1416 / 4,
    ):
        self.image = self.image.convert("L")
        if not stylish:
            return
        data = np.array(self.image)
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

        self.image = Image.fromarray(
            (255 * (dx * uni_x + dy * uni_y + dz * uni_z)).clip(0, 255).astype("uint8")
        )


def draw_figure(rules: "dict", path: str):
    # TODO apply rules to draw shapes (DONE)
    # TODO control their line weight and curves (MANUALLY)

    # TODO add various backgrounds and noise (HOW?)
    figure = Figure(rules, random_seed=0)
    figure.draw(n_redraw=100)
    figure.save(path)


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
    for i, sample in enumerate(samples):
        draw_figure(sample, f"dataset/pictures/{i}_PIL.png")


if __name__ == "__main__":
    main()

"draw geometry shapes according to generated rules"
import os, json, sys
from typing import Any
from PIL import Image, ImageDraw
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import data_args


class Figure(object):
    def __init__(
        self,
        rules: "list[dict[str, Any]]",
        random_seed: int,
        width: int = 1024,
        height: int = 1024,
    ) -> None:
        self.rules = rules
        self.random_seed = random_seed
        self.width = width
        self.height = height
        self.image = Image.new(mode="RGB", size=(width, height), color="white")
        self.canvas = ImageDraw.Draw(self.image, mode="RGB")
        random.seed(self.random_seed)

    def draw(self, n_pictures: int):
        for _ in range(n_pictures):
            for rule in self.rules:
                self.handle(rule)

    def __translate(self, point: tuple) -> tuple:
        return (self.width * point[0], self.height * point[1])

    def add_noise(self):
        pass

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

    def handle(self, rule: "dict[str, Any]", randomize: bool = True):
        match rule["type"]:
            case "polygon":
                points: list = [self.__translate(point) for point in rule["points"]]
                assert (
                    len(points) >= 3
                ), "There should be more than 3 points within a polygon."
                line_width: int = (
                    random.randint(0, 2) if randomize else 0
                )  # TODO: Improve the randomness
                for index in range(len(points)):
                    self.canvas.line(
                        xy=(
                            points[index][0],
                            points[index][1],
                            points[(index + 1) % len(points)][0],
                            points[(index + 1) % len(points)][1],
                        ),
                        width=line_width,
                        fill="black",
                    )
            case "line":
                points: list = [self.__translate(point) for point in rule["points"]]
                leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)
                line_width: int = (
                    random.randint(0, 2) if randomize else 0
                )  # TODO: Improve the randomness
                self.canvas.line(
                    xy=(
                        leftwise_endpoint[0],
                        leftwise_endpoint[1],
                        rightwise_endpoint[0],
                        rightwise_endpoint[1],
                    ),
                    width=line_width,
                    fill="black",
                )

            case "ray":
                points: list = [self.__translate(point) for point in rule["points"]]
                leftwise_endpoint, rightwise_endpoint = self.__line_extend(points)
                line_width: int = (
                    random.randint(0, 2) if randomize else 0
                )  # TODO: Improve the randomness
                farwise = (
                    leftwise_endpoint
                    if points[0][0] > points[1][0]
                    else rightwise_endpoint
                )

                self.canvas.line(
                    xy=(points[0][0], points[0][1], farwise[0], farwise[1]),
                    width=line_width,
                    fill="black",
                )
            case "segment":
                points: list = [self.__translate(point) for point in rule["points"]]
                line_width: int = (
                    random.randint(0, 2) if randomize else 0
                )  # TODO: Improve the randomness
                self.canvas.line(
                    xy=(
                        points[0][0],
                        points[0][1],
                        points[1][0],
                        points[1][1],
                    ),
                    width=line_width,
                    fill="black",
                )
            case "ellipse":
                ellipse_x, ellipse_y = self.__translate(rule["center"])
                major = int(rule["major_axis"] * self.width)
                minor = int(rule["minor_axis"] * self.height)
                alpha = rule["rotation"]

                ellipse_x -= major // 2
                ellipse_y -= major // 2

                im_ellipse = Image.new("RGBA", (major, major), (255, 255, 255, 0))
                cvs_ellipse = ImageDraw.Draw(im_ellipse, "RGBA")

                cvs_ellipse.ellipse(
                    (0, major // 2 - minor // 2, major, major // 2 + minor // 2),
                    fill=(255, 255, 255, 0),
                    outline="black",
                    width=4 + random.randint(-1, 3) if randomize else 3,
                )  # original ellipse

                rotated = im_ellipse.rotate(alpha * 180 / 3.1416)  # should be enough
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

            case _:
                raise ValueError(f"{rule['type']} is not any given rule.")

    def save(self, path: str):
        self.image.save(fp=path)


def draw_figure(rules: "list[dict[str, Any]]", index: int, path: str):
    # TODO apply rules to draw shapes
    # TODO control their line weight and curves

    # TODO add various backgrounds and noise
    figure = Figure(rules, random_seed=0)
    figure.draw(n_pictures=1)
    figure.save(path)


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
    for i, sample in enumerate(samples):
        draw_figure(sample, i, f"dataset/pictures/{i}.png")


if __name__ == "__main__":
    main()

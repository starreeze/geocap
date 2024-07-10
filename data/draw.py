"draw geometry shapes according to generated rules"
import os, json, sys
from typing import Any
from PIL import Image, ImageDraw

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import data_args


class Figure(object):
    def __init__(
        self,
        rules: "list[dict[str, Any]]",
        random_seed: int,
        width: int = 256,
        height: int = 256,
    ) -> None:
        self.rules = rules
        self.random_seed = random_seed
        self.canvas = Image.new(mode="I", size=(width, height), color="white")

    def draw(self, n_pictures: int):
        for _ in range(n_pictures):
            for rule in self.rules:
                self.handle(rule)

    def add_noise(self):
        pass

    def handle(self, rule: "dict[str, Any]"):
        match rule["type"]:
            case "Polygon":
                pass
            case "Line":
                pass
            case "Ellipse":
                pass

    def save(self, path: str):
        self.canvas.save(fp=path)


def draw_figure(rules: "list[dict[str, Any]]", index: int, path: str):
    # TODO apply rules to draw shapes
    # TODO control their line weight and curves

    # TODO add various backgrounds and noise
    figure = Figure(rules, random_seed=0)
    figure.draw(n_pictures=1)
    # figure.save(path)


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
    for i, sample in enumerate(samples):
        draw_figure(sample, i, f"dataset/pictures/{i}.png")


if __name__ == "__main__":
    main()

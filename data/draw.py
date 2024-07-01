"draw geometry shapes according to generated rules"
import os, json
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse
from common.args import data_args


def draw_figure(rules: list[dict[str, Any]], index: int):
    # TODO apply rules to draw shapes
    # TODO control their line weight and curves
    # NOTE below is an example for drawing shapes
    fig, ax = plt.subplots()

    # Draw polygon
    triangle = Polygon(((0.1, 0.3), (0.4, 0.8), (0.7, 0.3)), closed=True, edgecolor="black", fill=False)
    ax.add_patch(triangle)

    # Draw circle
    circle = Circle((0.5, 0.1), 0.1, edgecolor="black", fill=False)
    ax.add_patch(circle)
    ax.plot(0.5, 0.1, "ko")  # Center point of circle

    # Draw square (rotated 30 degrees manually)
    square_coords = [(0.7, 0.5), (0.9, 0.7), (1.1, 0.5), (0.9, 0.3)]
    square = Polygon(square_coords, closed=True, edgecolor="black", fill=False)
    ax.add_patch(square)

    # Draw ellipse
    ellipse = Ellipse((0.95, 0.4), 0.3, 0.1, angle=30, edgecolor="black", fill=False)
    ax.add_patch(ellipse)

    # TODO add various backgrounds and noise

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    ax.set_aspect("equal", "box")
    plt.axis("off")
    plt.savefig(os.path.join(data_args.figure_dir, f"{index}.jpg"))
    plt.close(fig)
    # plt.show()


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
    for i, sample in enumerate(samples):
        draw_figure(sample, i)


if __name__ == "__main__":
    main()

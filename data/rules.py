"generate rules for producing geometry shapes"
import json, json_fix
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Any
from common.args import data_args
import numpy as np


@dataclass
class GSRule(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        pass

    # __json__ = to_dict


@dataclass
class Polygon(GSRule):
    points: list[tuple[float, float]]

    def to_dict(self) -> dict[str, Any]:
        return {"type": "polygon"} | asdict(self)


@dataclass
class Line(GSRule):
    type: str  # line, segment, ray
    points: list[tuple[float, float]]  # two points determine the line

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Ellipse(GSRule):
    center: tuple[float, float]
    major_axis: float
    minor_axis: float
    rotation: float  # e.g., pi/3

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ellipse"} | asdict(self)


# TODO: add more types
@dataclass
class Spiral(GSRule):
    # Archimedean spiral  r = a + b*theta
    initial_radius: float  # a
    growth_rate: float  # b
    max_theta: float  # max theta to plot, e.g. 4*pi means the spiral will make 2 turns

    def to_dict(self) -> dict[str, Any]:
        return {"type": "spiral"} | asdict(self)


def generate_rules(num_samples=1000, max_num_shapes=5) -> list[list[GSRule]]:
    """
    Generate random rules across different types and shapes. Then mix them together.
    Returns: a list of samples where each consists a list of shapes.
    """
    # NOTE: all the shape should be in [0, 1] for both x and y
    # TODO: generate different types and shapes
    from numpy.random import randint, uniform

    def generate_random_polygon() -> Polygon:
        num_points = randint(3, 7)  # at least 3 points
        points = [(uniform(0, 1), uniform(0, 1)) for _ in range(num_points)]
        return Polygon(points)

    def generate_random_line() -> Line:
        point1 = (uniform(0, 1), uniform(0, 1))
        point2 = (uniform(0, 1), uniform(0, 1))
        line_type = np.random.choice(["line", "segment", "ray"])
        return Line(type=line_type, points=[point1, point2])

    def generate_random_ellipse() -> Ellipse:
        center = (uniform(0, 1), uniform(0, 1))
        major_axis = uniform(0, 1)
        minor_axis = uniform(0, major_axis)
        rotation = uniform(0, np.pi)
        return Ellipse(center, major_axis, minor_axis, rotation)

    def generate_random_spiral() -> Spiral:
        initial_radius = uniform(0, 1)
        growth_rate = uniform(0, 1)
        max_theta = uniform(2 * np.pi, 20 * np.pi)  # at least 1 full turn
        return Spiral(initial_radius, growth_rate, max_theta)

    # TODO: how to mix them to form a sample
    results = []
    shape_generators = [
        generate_random_polygon,
        generate_random_line,
        generate_random_ellipse,
        generate_random_spiral,
    ]

    for _ in range(num_samples):
        num_shapes = randint(1, max_num_shapes)
        sample = [np.random.choice(shape_generators)().to_dict() for _ in range(num_shapes)]
        results.append(sample)

    assert len(results) == num_samples
    return results


def save_rules(rules: list[list[GSRule]], output_file: str):
    with open(output_file, "w") as f:
        json.dump(rules, f, default=vars)


def main():
    samples = generate_rules(data_args.num_basic_geo_samples)
    save_rules(samples, data_args.rules_path)


if __name__ == "__main__":
    main()

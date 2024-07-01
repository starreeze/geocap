"generate rules for producing geometry shapes"
import os, json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Any
from common.args import data_args


@dataclass
class GSRule(ABC):
    @abstractmethod
    def __dict__(self) -> dict[str, Any]: ...


@dataclass
class Polygon(GSRule):
    points: list[tuple[float, float]]

    def __dict__(self) -> dict[str, Any]:
        return {"type": "polygon"} | asdict(self)


@dataclass
class Line(GSRule):
    type: str  # line, segment, ray
    points: list[tuple[float, float]]  # two points determine the line

    def __dict__(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Ellipse(GSRule):
    center: tuple[float, float]
    major_axis: float
    minor_axis: float
    rotation: float  # e.g., pi/3

    def __dict__(self) -> dict[str, Any]:
        return {"type": "ellipse"} | asdict(self)


# TODO: add more types


def generate_rules(num_samples=1000) -> list[list[GSRule]]:
    """
    Generate random rules across different types and shapes. Then mix them together.
    Returns: a list of samples where each consists a list of shapes.
    """
    # NOTE: all the shape should be in [0, 1] for both x and y
    # TODO: generate different types and shapes
    # TODO: how to mix them to form a sample

    results: list[list[GSRule]] = [[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] for _ in range(num_samples)]

    assert len(results) == num_samples
    return results


def save_rules(rules: list[list[GSRule]], output_file: str):
    with open(output_file, "w") as f:
        json.dump(rules, f)


def main():
    samples = generate_rules(data_args.num_basic_gep_samples)
    save_rules(samples, data_args.rules_path)


if __name__ == "__main__":
    main()

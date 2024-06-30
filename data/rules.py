"generate rules for producing geometry shapes"
import json
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from typing import Any


@dataclass
class GSRule(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


@dataclass
class Polygon(GSRule):
    points: list[tuple[float, float]]
    to_dict = lambda self: {"type": "polygon"} | asdict(self)


@dataclass
class Line(GSRule):
    type: str  # line, segment, ray
    points: list[tuple[float, float]]
    to_dict = lambda self: asdict(self)


@dataclass
class Ellipse(GSRule):
    center: tuple[float, float]
    major_axis: float
    minor_axis: float
    rotation: float  # e.g., pi/3
    to_dict = lambda self: {"type": "ellipse"} | asdict(self)


def generate_rules(num_samples=1000) -> list[GSRule]:
    "generate random rules across different types and shapes"
    # NOTE: all the shape should be in [0, 1] for both x and y
    # TODO: this is a placeholder for now
    return [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]) for _ in range(num_samples)]


def save_rules(rules: list[GSRule], output_file="dataset/gs_rules.json"):
    with open(output_file, "w") as f:
        json.dump([rule.to_dict() for rule in rules], f, indent=2)


if __name__ == "__main__":
    rules = generate_rules(1)
    save_rules(rules)

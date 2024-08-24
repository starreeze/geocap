from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging, os
from rich.logging import RichHandler
from typing import Any, cast

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")


@dataclass
class DataArgs:
    rules_path: str = field(default="dataset/rules.json")
    figure_dir: str = field(default="dataset/geo-shapes")
    captions_path: str = field(default="dataset/captions.json")
    num_basic_geo_samples: int = field(default=1000)

    llava_data_path: str = field(default="dataset/llava-data.json")


@dataclass
class RunArgs:
    module: str = field(default="")
    action: str = field(default="main")

    num_workers: int = field(default=32)


@dataclass
class RuleArgs:
    # max number of shapes in each sample
    max_num_shapes: int = field(default=10)

    # levels of shape generation
    polygon_shape_level: int = field(default=3)
    line_shape_level: int = field(default=1)
    ellipse_shape_level: int = field(default=4)
    spiral_shape_level: int = field(default=3)

    # levels of polygon relation
    polygon_tangent_line_level: int = field(default=1)
    polygon_symmetric_level: int = field(default=1)
    polygon_similar_level: int = field(default=1)
    polygon_shared_edge_level: int = field(default=3)
    polygon_circumscribed_circle_of_triangle_level: int = field(default=2)
    polygon_inscribed_circle_level: int = field(default=2)
    polygon_circumscribed_circle_of_rectangle_level: int = field(default=2)
    polygon_diagonal_level: int = field(default=1)

    # levels of line relation
    line_parallel_level: int = field(default=1)
    line_tangent_line_level: int = field(default=2)
    line_axis_of_ellipse_level: int = field(default=2)

    # levels of ellipse relation
    ellipse_tangent_line_level: int = field(default=1)
    ellipse_tangent_circle_level: int = field(default=2)
    ellipse_concentric_level: int = field(default=3)
    ellipse_circumscribed_level: int = field(default=3)
    ellipse_inscribed_level: int = field(default=3)


@dataclass
class DrawArgs:
    serial_version: bool = field(default=False)
    backend: "str" = field(default="plt")
    random_seed: None | int = field(default=None)
    randomize: bool = field(default=True)
    size: "tuple[float, float]" = field(default=(6.4, 6.4))
    dpi: int = field(default=100)
    line_weight: int = field(default=4)
    xkcd: bool = field(default=True)
    color: None | tuple = field(default=None)
    n_white_line: None | int = field(default=None)
    Gaussian_mean: int = field(default=0)
    Gaussian_var: float = field(default=10)
    Perlin_lattice: int = field(default=20)
    Perlin_power: float = field(default=16)
    Perlin_bias: float = field(default=-16)
    stylish: bool = field(default=False)


data_args, run_args, rule_args, draw_args = HfArgumentParser([DataArgs, RunArgs, RuleArgs, DrawArgs]).parse_args_into_dataclasses()  # type: ignore
data_args = cast(DataArgs, data_args)
run_args = cast(RunArgs, run_args)
rule_args = cast(DataArgs, rule_args)
draw_args = cast(DrawArgs, draw_args)

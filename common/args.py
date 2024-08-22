from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging, os
from rich.logging import RichHandler
from typing import cast, Any

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
class DrawArgs:
    rules: "list[dict[str, Any]]" = field()
    random_seed: None | int = field()
    randomize: bool = field(default=True)
    size: "tuple[float, float]" = field(default=(6.4, 6.4))
    dpi: int = field(default=100)
    line_weight: int = field(default=4)
    xkcd: bool = field(default=False)
    color: None | tuple = field(default=None)
    n_white_line: None | int = field(default=None)
    Gaussian_mean: float = field(default=0)
    Gaussian_var: float = field(default=10)
    Perlin_lattice: int = field(default=20)
    Perlin_power: float = field(default=16)
    Perlin_bias: float = field(default=-16)
    stylish: bool = field(default=False)


data_args, run_args, draw_args = HfArgumentParser([DataArgs, RunArgs, DrawArgs]).parse_args_into_dataclasses()  # type: ignore
data_args = cast(DataArgs, data_args)
run_args = cast(RunArgs, run_args)
draw_args = cast(DrawArgs, draw_args)

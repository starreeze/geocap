import logging
import os
from dataclasses import dataclass, field
from typing import Literal, cast

from rich.logging import RichHandler
from transformers import HfArgumentParser


@dataclass
class DataArgs:
    rules_path: str = field(default="dataset/rules.json")
    figure_dir: str = field(default="dataset/geo-shapes")
    figure_name: str = field(default="{prefix}_{id:08d}.jpg")
    caption_dir: str = field(default="dataset")
    vqa_dir: str = field(default="dataset/vqa")
    stage: int = field(default=1)
    num_basic_geo_samples: int = field(default=100000)
    num_fossil_samples: int = field(default=3)
    llava_data_dir: str = field(default="dataset/llava")

    # some placeholder to be filled AFTER parsing args
    caption_path: str = field(default="")
    figure_prefix: str = field(default="")
    llava_data_path: str = field(default="")


@dataclass
class RunArgs:
    module: str = field(default="")
    action: str = field(default="main")
    log_level: str = field(default="INFO")
    num_workers: int = field(default=32)
    progress_bar: bool = field(default=True)
    start_pos: int = field(default=0)
    end_pos: int = field(default=100000)


@dataclass
class RuleArgs:
    output_fp_precision: int = field(default=4)

    """args for stage 1"""
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

    """args for stage 2"""
    prob_has_axial_filling: float = field(default=0.8)
    overlap_axial_and_poles_folds: bool = False


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


@dataclass
class CaptionArgs:
    caption_batchsize: int = field(default=4)
    caption_llm: str = field(default="llama31-8")
    numeric_ratio: float = field(default=0)


@dataclass
class VQAArgs:
    vqa_batchsize: int = field(default=4)
    vqa_llm: str = field(default="qwen25-7")
    vqa_prompts_dir: str = field(default="data/vqa/prompts")
    max_q_ip: int = field(default=3, metadata={"help": "maximum number of questions per image per perspective"})
    vqa_digits: int = field(default=2, metadata={"help": "number of digits for the answer"})
    nrel_q_prob: float = field(default=0.3, metadata={"help": "probability of no-relation questions"})
    size_diff: float = field(
        default=0.15,
        metadata={"help": "ratio of the difference of the correct answer and the other choices for size questions"},
    )
    area_type_t: float = field(
        default=0.05, metadata={"help": "tolerate threshold for area difference to be considered"}
    )
    location_type_t: float = field(
        default=0.1, metadata={"help": "tolerate threshold for location difference to be considered"}
    )


data_args, run_args, rule_args, draw_args, caption_args, vqa_args = HfArgumentParser(
    [DataArgs, RunArgs, RuleArgs, DrawArgs, CaptionArgs, VQAArgs]  # type: ignore
).parse_args_into_dataclasses()

data_args = cast(DataArgs, data_args)
run_args = cast(RunArgs, run_args)
rule_args = cast(RuleArgs, rule_args)
draw_args = cast(DrawArgs, draw_args)
caption_args = cast(CaptionArgs, caption_args)
vqa_args = cast(VQAArgs, vqa_args)

data_args.figure_prefix = (
    data_args.figure_prefix if data_args.figure_prefix else (draw_args.backend if draw_args.randomize else "pure")
)
data_args.caption_path = (
    data_args.caption_path
    if data_args.caption_path
    else os.path.join(data_args.caption_dir, f"n{caption_args.numeric_ratio}_{run_args.end_pos//1000:03d}k.jsonl")
)
data_args.llava_data_path = (
    data_args.llava_data_path
    if data_args.llava_data_path
    else os.path.join(data_args.llava_data_dir, f"{data_args.figure_prefix}_n{caption_args.numeric_ratio}.json")
)

logging.basicConfig(level=run_args.log_level, format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")

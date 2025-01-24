import logging
import math
import os
import sys
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from rich.logging import RichHandler
from transformers import HfArgumentParser


@dataclass
class DataArgs:
    rules_path: str = field(default="dataset/rules.json")
    figure_dir: str = field(default="dataset/figures")
    figure_name: str = field(default="{prefix}_{id:08d}.jpg")
    caption_dir: str = field(default="dataset")
    vqa_question_dir: str = field(default="dataset/vqa")
    vqa_output_dir: str = field(default="results")
    stage: int = field(default=1)
    # set num_samples for each num_shapes
    # num_samples_per_num_shapes[i]: number of samples for num_shapes=(min_num_shapes + i)
    num_samples_per_num_shapes: list[int] = field(default_factory=lambda: [10, 10, 10])
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
    end_pos: int = field(default=sys.maxsize)
    api_key_file: str = field(default="api_key.yaml")


@dataclass
class RuleArgs:
    output_fp_precision: int = field(default=4)

    """args for stage 1"""
    min_num_shapes: int = field(default=2)
    in_canvas_area_thres: float = field(default=0.8)
    # levels of shape generation
    polygon_shape_level: int = field(default=5)
    line_shape_level: int = field(default=2)
    ellipse_shape_level: int = field(default=3)
    spiral_shape_level: int = field(default=1)

    # numerical params for shapes
    polygon_points_min_distance: float = field(default=0.01)
    rectangle_ratio_thres: list[float] = field(default_factory=lambda: [1.2, 4.0])
    general_quadrilateral_angle_thres: float = field(default=0.3)
    general_triangle_angle_thres: float = field(default=0.3)

    line_min_length: float = field(default=0.2)
    line_max_length: float = field(default=0.5)

    ellipse_ratio_thres: list[float] = field(default_factory=lambda: [1.2, 4.0])

    # levels of polygon relation
    polygon_tangent_line_level: int = field(default=2)
    polygon_symmetric_level: int = field(default=1)
    polygon_similar_level: int = field(default=1)
    polygon_shared_edge_level: int = field(default=2)
    polygon_circumscribed_circle_of_triangle_level: int = field(default=2)
    polygon_inscribed_circle_level: int = field(default=2)
    polygon_circumscribed_circle_of_rectangle_level: int = field(default=2)
    polygon_circumscribed_circle_of_square_level: int = field(default=2)
    polygon_diagonal_level: int = field(default=1)

    # levels of line relation
    line_parallel_level: int = field(default=1)
    line_tangent_line_level: int = field(default=2)
    line_axis_of_ellipse_level: int = field(default=2)

    # levels of ellipse relation
    ellipse_tangent_line_level: int = field(default=3)
    ellipse_tangent_circle_level: int = field(default=3)
    ellipse_concentric_level: int = field(default=2)
    ellipse_circumscribed_level: int = field(default=3)
    ellipse_inscribed_level: int = field(default=3)

    """args for stage 2"""
    prob_has_axial_filling: float = field(default=0.8)
    overlap_axial_and_poles_folds: bool = False


@dataclass
class DrawArgs:
    serial_version: bool = field(default=False)
    fig_id_start: int = field(default=0)
    backend: "str" = field(default="plt")
    random_seed: None | int = field(default=None)
    randomize: bool = field(default=True)
    size: "list[float]" = field(default_factory=lambda: [6.4, 6.4])
    dpi: int = field(default=100)
    line_weight: int = field(default=4)
    line_style: Literal["none", "gradient", "xkcd"] = field(default="none")
    color: list[float] = field(default_factory=lambda: [])
    n_white_line: None | int = field(default=None)
    white_line_range: float = field(default=0.25)
    Gaussian_mean: int = field(default=0)
    Gaussian_var: float = field(default=10)
    Gaussian_proba: float = field(default=1)
    Perlin_lattice: int = field(default=20)
    Perlin_power: float = field(default=16)
    Perlin_bias: float = field(default=-16)
    Perlin_proba: float = field(default=1)
    inline_noise: bool = field(default=True)
    stylish: bool = field(default=False)
    stylish_alpha: float = field(default=3.1416 / 4)
    stylish_depth: int = field(default=10)
    stylish_height: float = field(default=3.1416 / 2.2)


@dataclass
class CaptionArgs:
    caption_batchsize: int = field(default=4)
    caption_llm: str = field(default="llama31-8")
    numeric_ratio: float = field(default=0)
    debug_option: str = field(default="")


@dataclass
class VQAArgs:
    perspectives: list[str] = field(
        default_factory=lambda: [
            "existence",
            "counting",
            "size",
            "location",
            "reference",
            "relation",
        ]
    )
    # llm generator
    vqa_batchsize: int = field(default=4)
    vqa_llm: str = field(default="qwen25-7")
    vqa_prompts_dir: str = field(default="data/vqa/prompts")
    # rule generator
    max_q_ip: int = field(
        default=3,
        metadata={"help": "maximum number of questions per image per perspective"},
    )
    vqa_digits: int = field(
        default=2, metadata={"help": "number of digits for the answer"}
    )
    nrel_q_prob: float = field(
        default=0.3, metadata={"help": "probability of no-relation questions"}
    )
    gt_choice_w: list[float] = field(
        default_factory=lambda: [0.05, 0.15, 0.25, 0.55],
        metadata={
            "help": "weight of the correct answer in the 4 choices; "
            "we need this to balance the answer distribution "
            "since smaller values are naturally more likely to be correct"
        },
    )
    size_diff: float = field(
        default=0.15,
        metadata={
            "help": "ratio of the difference of the correct answer and the other choices for size questions"
        },
    )
    area_type_t: float = field(
        default=0.02,
        metadata={"help": "tolerate threshold for area difference to be considered"},
    )
    location_type_t: float = field(
        default=0.03,
        metadata={
            "help": "tolerate threshold for location difference to be considered"
        },
    )
    # evaluation
    eval_model: str = field(
        default="llava-7b",
        metadata={
            "help": "model name for evaluation. Naming convention: {model_name}-{model_size}"
        },
    )
    eval_batchsize: int = field(default=4)
    eval_inst: str = field(
        default="Please directly answer A, B, C or D and nothing else."
    )

    distinguish_threshold_of_relative_direction: float = field(default=0.04)
    deviation_threshold_of_relative_direction: float = field(default=math.pi / 9)
    exclusiv_deviation_threshold_of_relative_direction: float = field(
        default=math.pi / 5
    )
    relative_direction_text_and_vector_dict: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )
    distinguish_threshold_of_absolute_direction: float = field(default=0.1)
    absolute_direction_text_and_box_dict: dict[
        str, tuple[tuple[float, float], tuple[float, float]]
    ] = field(default_factory=dict)
    inclusiv_overlapping_threshold_of_absolute_direction: float = field(default=0.8)

    def __post_init__(self):
        object.__setattr__(
            self,
            "relative_direction_text_and_vector_dict",
            MappingProxyType(
                {
                    "directly to the top of": (0, 1),
                    "directly to the bottom of": (0, -1),
                    "directly to the left of": (-1, 0),
                    "directly to the right of": (1, 0),
                    "to the upper left of": (-math.sqrt(2), math.sqrt(2)),
                    "to the upper right of": (math.sqrt(2), math.sqrt(2)),
                    "to the lower left of": (-math.sqrt(2), -math.sqrt(2)),
                    "to the lower right of": (math.sqrt(2), -math.sqrt(2)),
                }
            ),
        )
        object.__setattr__(
            self,
            "absolute_direction_text_and_box_dict",
            MappingProxyType(
                {
                    "in the upper half of the image": (
                        (-1, 2),
                        (2, 0.5 + self.distinguish_threshold_of_absolute_direction),
                    ),
                    "in the lower half of the image": (
                        (-1, 0.5 - self.distinguish_threshold_of_absolute_direction),
                        (2, -1),
                    ),
                    "in the left half of the image": (
                        (-1, 2),
                        (0.5 - self.distinguish_threshold_of_absolute_direction, -1),
                    ),
                    "in the right half of the image": (
                        (0.5 + self.distinguish_threshold_of_absolute_direction, 2),
                        (2, -1),
                    ),
                    "in the top left quarter of the image": (
                        (-1, 2),
                        (
                            0.5 - self.distinguish_threshold_of_absolute_direction,
                            0.5 + self.distinguish_threshold_of_absolute_direction,
                        ),
                    ),
                    "in the top right quarter of the image": (
                        (0.5 + self.distinguish_threshold_of_absolute_direction, 2),
                        (2, 0.5 + self.distinguish_threshold_of_absolute_direction),
                    ),
                    "in the bottom left quarter of the image": (
                        (-1, 0.5 - self.distinguish_threshold_of_absolute_direction),
                        (0.5 - self.distinguish_threshold_of_absolute_direction, -1),
                    ),
                    "in the bottom right quarter of the image": (
                        (
                            0.5 + self.distinguish_threshold_of_absolute_direction,
                            0.5 - self.distinguish_threshold_of_absolute_direction,
                        ),
                        (2, -1),
                    ),
                }
            ),
        )


@dataclass
class FeatureRecognizeArgs:
    houghcircle_params: dict[str, float] = field(
        default_factory=lambda: {
            "dp": 1.5,
            "minDist": 100,
            "param1": 150,
            "param2": 0.5,
        },
        metadata={
            "help": "parameters for cv2.HoughCircles: dp, minDist, param1, param2"
        },
    )
    volution_thres: float = field(
        default=0.85, metadata={"help": "threshold for volution detection"}
    )

    fossil_data_path: str = field(default="dataset/common")
    desc_llm: str = field(default="qwen25-14")
    desc_prompt_dir: str = field(default="feat_recognize/prompt.txt")
    save_data_path: str = field(default="dataset/")


(
    data_args,
    run_args,
    rule_args,
    draw_args,
    caption_args,
    vqa_args,
    feat_recog_args,
) = HfArgumentParser(
    [DataArgs, RunArgs, RuleArgs, DrawArgs, CaptionArgs, VQAArgs, FeatureRecognizeArgs]  # type: ignore
).parse_args_into_dataclasses()

data_args = cast(DataArgs, data_args)
run_args = cast(RunArgs, run_args)
rule_args = cast(RuleArgs, rule_args)
draw_args = cast(DrawArgs, draw_args)
caption_args = cast(CaptionArgs, caption_args)
vqa_args = cast(VQAArgs, vqa_args)
feat_recog_args = cast(FeatureRecognizeArgs, feat_recog_args)

data_args.figure_prefix = (
    data_args.figure_prefix
    if data_args.figure_prefix
    else (draw_args.backend if draw_args.randomize else "pure")
)
data_args.caption_path = (
    data_args.caption_path
    if data_args.caption_path
    else os.path.join(
        data_args.caption_dir,
        f"n{caption_args.numeric_ratio}_{run_args.end_pos//1000:03d}k.jsonl",
    )
)
data_args.llava_data_path = (
    data_args.llava_data_path
    if data_args.llava_data_path
    else os.path.join(
        data_args.llava_data_dir,
        f"{data_args.figure_prefix}_n{caption_args.numeric_ratio}.json",
    )
)
assert vqa_args.size_diff < 0.2, "size_diff should be less than 0.2"
run_args.log_level = run_args.log_level.upper()
logging.basicConfig(
    level=run_args.log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger("rich")
logger.setLevel(run_args.log_level)

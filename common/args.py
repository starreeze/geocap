from dataclasses import dataclass, field
from transformers import HfArgumentParser
import logging, os
from rich.logging import RichHandler
from typing import cast

logging.basicConfig(level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()])
logger = logging.getLogger("rich")


@dataclass
class DataArgs:
    rules_path: str = field(default="dataset/rules.json")
    figure_dir: str = field(default="dataset/geo-shapes")
    captions_path: str = field(default="dataset/captions.json")
    num_basic_gep_samples: int = field(default=1000)


@dataclass
class RunArgs:
    module: str = field(default="")
    action: str = field(default="main")


data_args, run_args = HfArgumentParser([DataArgs, RunArgs]).parse_args_into_dataclasses()  # type: ignore
data_args = cast(DataArgs, data_args)
run_args = cast(RunArgs, run_args)

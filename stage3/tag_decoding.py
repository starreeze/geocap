import importlib
import json
import math
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common.args import data_args, logger, run_args, vqa_args
from common.vllm.base import GenerateModelBase
from data.rule.utils import round_floats

Model = importlib.import_module(f"common.vllm.{vqa_args.eval_model.split('-')[0]}").GenerateModel


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()

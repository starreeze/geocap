# -*- coding: utf-8 -*-
# @Date    : 2024-12-09 11:15:38
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from eval.base import GenerateModelBase
import argparse
from PIL import Image
import sys

sys.path.append("/MiniGPT")
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


class GenerateModel(GenerateModelBase):
    def __init__(self):
        args = argparse.Namespace(cfg_path="/MiniGPT/eval_configs/minigptv2_eval.yaml", gpu_id=0, options=[])
        cfg = Config(args)
        self.device = f"cuda:{args.gpu_id}"
        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        self.model_cls = registry.get_model_class(model_config.arch)
        self.model = self.model_cls.from_config(model_config).to(self.device)
        self.vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(self.vis_processor_cfg.name).from_config(
            self.vis_processor_cfg
        )
        self.model.eval()

    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        assert len(image_paths) == len(prompts)
        res = []
        for image_path, prompt in zip(image_paths, prompts):
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)
            image = image.unsqueeze(0)
            prompt = [f"[INST] <Img><ImageHere></Img> {prompt}[/INST]"]
            answers = self.model.generate(image, prompt, max_new_tokens=10, do_sample=False)
            res.append(answers[0].strip())

        return res

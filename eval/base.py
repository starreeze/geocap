# -*- coding: utf-8 -*-
# @Date    : 2024-12-10 09:34:27
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"protocol definition for the generation model"

from abc import ABC, abstractmethod


class GenerateModelBase(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, image_paths: list[str], prompts: list[str]) -> list[str]:
        """
        generate the responses of the model on the given image paths and prompts
        Args:
            image_paths: list[str], the paths of the input images in a batch
            prompts: list[str], the user prompts for the model in a batch
        Returns:
            list[str]: the raw responses of the model
            (no need to do post-processing, but prompts should not be included)
        """
        pass

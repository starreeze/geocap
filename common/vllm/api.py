# -*- coding: utf-8 -*-
# @Date    : 2024-12-19 10:31:46
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from common.args import vqa_args
from common.llm import APIGenerator

from .base import GenerateModelBase


class GenerateModel(GenerateModelBase):
    def __init__(self):
        self.generator = APIGenerator(vqa_args.eval_model.split("-", 1)[1])

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
        responses = []
        for image_path, prompt in zip(image_paths, prompts):
            response = self.generator.get_one_response([("image", image_path), ("text", prompt)])
            responses.append(response)
        return responses


# After finishing the implementation, you can first test our implementation using test data
# Use python -m eval.{file_name} to run this file
if __name__ == "__main__":
    model = GenerateModel()
    res = model.generate(
        ["test-dataset/geo-shapes/plt_00000000.jpg"],
        [
            "What is the number of triangles in the image?\nA. 0\nB. 1\nC. 2\nD. 3\n"
            "Please directly answer A, B, C or D and nothing else."
        ],
    )
    print(res)

# Then run the following command to evaluate the model
# ./run -m eval.evaluate --eval_model {model_name}-{model_size} --eval_batchsize 4
# and record the results in the table

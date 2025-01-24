# -*- coding: utf-8 -*-
# @Date    : 2024-12-08 10:52:07
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import json
import random
from typing import Any

from tqdm import tqdm

from common.args import data_args, logger, vqa_args
from common.llm import LLMGenerator, generator_mapping, model_path_mapping
from data.vqa.base import GeneratorBase


class LLMQAGenerator(GeneratorBase):
    llm_generator: LLMGenerator | None = None

    def __init__(
        self,
        rules: list[dict[str, Any]],
        sys_prompt="You are a helpful assistant that always responds in json.",
    ):
        super().__init__(rules)
        if self.llm_generator is None:
            with open(data_args.caption_path) as f:
                self.captions = [json.loads(line)["output"] for line in f]
            model_name, model_id = vqa_args.vqa_llm.split("-", 1)
            model_path = model_path_mapping[model_name].format(model_id)
            self.__class__.llm_generator = generator_mapping[model_name](model_path)
        self.sys_prompt = sys_prompt

    def __call__(self, task_prompt: str) -> list[dict[str, Any]]:
        assert self.llm_generator is not None
        qa_pairs: list[dict[str, Any]] = []
        inputs = [
            [
                {"role": "system", "content": self.sys_prompt},
                {
                    "role": "user",
                    "content": task_prompt.replace("{caption}", caption).replace(
                        "{rules}", json.dumps(rules)
                    ),
                },
            ]
            for caption, rules in zip(self.captions, self.data)
        ]
        num_batches = (len(inputs) + vqa_args.vqa_batchsize - 1) // vqa_args.vqa_batchsize
        responses = self.llm_generator(inputs, vqa_args.vqa_batchsize)
        for i, batch in tqdm(enumerate(responses), total=num_batches):
            for j, response in enumerate(batch):
                image_id = i * vqa_args.vqa_batchsize + j
                qas: list[dict[str, Any]] | None = self.extract_first_json_array(response)
                if qas is None:
                    logger.warning(f"No JSON list found in response {response}. Input: {inputs[image_id]}")
                    continue
                for qa in qas[: vqa_args.max_q_ip]:
                    qa: dict[str, Any] = {"image_id": image_id} | qa
                    if "none" in qa["choices"][-1]:  # shuffle the first three choices
                        first_three_choices = qa["choices"][:3]
                        random.shuffle(first_three_choices)
                        qa["choices"] = first_three_choices + [qa["choices"][-1]]
                    else:
                        random.shuffle(qa["choices"])
                    qa_pairs.append(qa)
        return qa_pairs

    @staticmethod
    def extract_first_json_array(text: str) -> list | None:
        """Extract the first valid JSON array from text."""
        stack = []
        for i, char in enumerate(text):
            if char == "[":
                stack.append(i)
            elif char == "]" and stack:
                start = stack.pop()
                if not stack:  # We found a complete top-level array
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        continue
        return None

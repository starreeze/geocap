# -*- coding: utf-8 -*-
# @Date    : 2024-12-08 10:52:07
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import json
import random
from typing import Any

from tqdm import tqdm

from common.args import logger, vqa_args
from common.llm import LLMGenerator, generator_mapping, model_path_mapping


class LLMQAGenerator:
    llm_generator: LLMGenerator | None = None

    def __init__(
        self,
        task_prompt: str,
        sys_prompt="You are a helpful assistant that always responds in json.",
    ):
        if self.llm_generator is None:
            model_name, model_size = vqa_args.vqa_llm.split("-")
            model_path = model_path_mapping[model_name].format(size=model_size)
            self.__class__.llm_generator = generator_mapping[model_name](model_path)
        self.task_prompt = task_prompt
        self.sys_prompt = sys_prompt

    def __call__(self, captions: list[str]) -> list[dict[str, Any]]:
        assert self.llm_generator is not None
        qa_pairs: list[dict[str, Any]] = []
        inputs = [
            [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": self.task_prompt.replace("{caption}", caption)},
            ]
            for caption in captions
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

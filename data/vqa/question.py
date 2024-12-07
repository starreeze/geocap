# -*- coding: utf-8 -*-
# @Date    : 2024-12-03 11:18:24
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"construct VQA questions according to the captions and rules"

import json
import os
import random
from collections import Counter
from typing import Any

import numpy as np
from tqdm import tqdm

from common.args import data_args, logger, vqa_args
from common.llm import LLMGenerator, generator_mapping, model_path_mapping
from data.rule.shapes import GSRule


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


class RuleBasedQAGenerator:
    data: list[dict[str, Any]] = []  # [{shapes: [{type, center, box, area}], relations, counts}]
    total_shapes = [
        "line",
        "ellipse",
        "circle",
        "triangle",  # order cannot be changed; this will be indexed
        "quadrilateral",
        "pentagon",
        "hexagon",
        "rectangle",
        "square",
        "spiral",
    ]
    shape_hierarchy = {
        "ellipse": ["circle"],
        "rectangle": ["square"],
        "quadrilateral": ["rectangle", "square"],
    }
    total_relations = [
        "tangent",
        "parallel",
        "similar",
        "symmetric",
        "circumscribed",
        "inscribed",
        "shared edge",
        "diagonal",
        "major axis",
        "minor axis",
        "diameter",
        "concentric",
    ]

    @staticmethod
    def get_relation(relation: str, shape_1: GSRule, shape_2: GSRule) -> str:
        # TODO: refine relations
        if relation == "internal tangent":
            return "inscribed"
        if relation == "external tangent":
            return "circumscribed"
        return relation

    @classmethod
    def get_type(cls, shape_dict: dict[str, Any]) -> str:
        special_info = shape_dict.get("special_info", "").strip(" .").split(" ")[-1]
        if special_info:
            return special_info
        if shape_dict["type"] in ["segment", "ray"]:
            return "line"
        if shape_dict["type"] == "polygon":
            return cls.total_shapes[len(shape_dict["points"])]
        return shape_dict["type"]

    def __init__(self, rules: list[dict[str, Any]]):
        if self.data:
            return
        for figure in rules:
            info: dict[str, Any] = {"shapes": []}
            for shape_dict in figure["shapes"]:
                shape = GSRule.from_dict(shape_dict)
                shape_info = {
                    "type": self.get_type(shape_dict),
                    "center": shape.get_centroid(),
                    "box": shape.get_bbox(),
                    "area": shape.get_area(),
                }
                info["shapes"].append(shape_info)
                # TODO: merge relations
            info["relations"] = figure["relations"]
            info["counts"] = dict(Counter(shape["type"] for shape in info["shapes"]))
            self.data.append(info)

    def __call__(self, perspective: str) -> list[dict[str, Any]]:
        logger.info(f"Generating {perspective} questions")
        qa_pairs: list[dict[str, Any]] = []
        for i, figure in tqdm(enumerate(self.data), total=len(self.data)):
            for j, qa in enumerate(getattr(self, perspective)(figure)):
                qa_pairs.append({"image_id": i, "question_id": j} | qa)
        return qa_pairs

    @classmethod
    def counting(cls, figure: dict[str, Any]) -> list[dict[str, Any]]:
        qa_pairs: list[dict[str, Any]] = []
        counts: dict[str, int] = figure["counts"]

        # Step 1: Generate probability distribution based on counts
        total_count = sum(counts.values())
        prob_dist = {type: count / total_count for type, count in counts.items()}

        # Step 2: Generate questions based on observed shapes (without duplicates but weighted)
        types = list(counts.keys())
        probs = [prob_dist[type] for type in types]
        num_questions = min(len(types), vqa_args.max_q_ip - 1)
        selected_types = list(np.random.choice(types, size=num_questions, p=probs, replace=False))

        # Step 3: Add question for unobserved shape (count = 0)
        unobserved_types = [t for t in cls.total_shapes if t not in counts]
        zero_type = random.choice(unobserved_types) if unobserved_types else random.choice(types)

        # Step 4: Generate questions with choices
        for type in selected_types + [zero_type]:
            correct_answer = counts.get(type, 0)
            if correct_answer <= 3:
                choices = [0, 1, 2, 3]
            else:
                choices = [correct_answer - 3, correct_answer - 2, correct_answer - 1, correct_answer]

            if type in cls.shape_hierarchy:
                additional_desc = f", with {', '.join(cls.shape_hierarchy[type])} excluded"
            else:
                additional_desc = ""
            question = f"How many {type}(s) are there in the image{additional_desc}?"

            qa_pairs.append({"question": question, "choices": choices, "answer": correct_answer})
        return qa_pairs


def main():
    os.makedirs(data_args.vqa_dir, exist_ok=True)

    with open(data_args.caption_path) as f:
        captions = [json.loads(line)["output"] for line in f]
    with open(data_args.rules_path) as f:
        rules = json.load(f)

    # perspectives = ["existence", "counting", "size", "location", "type", "relation"]
    perspectives = ["counting"]

    for perspective in perspectives:
        prompt_file = os.path.join(vqa_args.vqa_prompts_dir, f"{perspective}.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file, "r") as f:
                prompt = f.read()
            qa_generator = LLMQAGenerator(prompt)
            qa_pairs = qa_generator(captions)
        else:
            qa_generator = RuleBasedQAGenerator(rules)
            qa_pairs = qa_generator(perspective)

        with open(os.path.join(data_args.vqa_dir, f"{perspective}.jsonl"), "w") as f:
            f.write("\n".join([json.dumps(qa) for qa in qa_pairs]))


if __name__ == "__main__":
    main()

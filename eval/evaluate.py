# -*- coding: utf-8 -*-
# @Date    : 2024-12-09 19:07:47
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import importlib
import json
import os
from typing import Any, Callable

from tqdm import tqdm

from common.args import data_args, logger, vqa_args

generate: Callable[[list[str], list[str]], str] = importlib.import_module(f"eval.{vqa_args.eval_model}").generate


def batched_evaluate(data: list[dict[str, Any]]) -> list[str]:
    batched_data = [data[i : i + vqa_args.vqa_batchsize] for i in range(0, len(data), vqa_args.vqa_batchsize)]
    answers = []
    for batch in tqdm(batched_data):
        image_paths = [
            os.path.join(
                data_args.figure_dir, data_args.figure_name.format(prefix=data_args.figure_prefix, id=item["image_id"])
            )
            for item in batch
        ]
        questions = [
            f"{item['question']}\n"
            + "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(item["choices"]))
            + f"\n{vqa_args.eval_inst}"
            for item in batch
        ]

        responses = generate(image_paths, questions)

        for resp in responses:
            found_answers = [letter for letter in "ABCD" if letter in resp.strip()]
            if len(found_answers) == 1:
                answers.append(found_answers[0])
            else:
                answers.append("-")
    return answers


def main():
    for perspective in vqa_args.perspectives:
        logger.info(f"Evaluating {perspective} on model {vqa_args.eval_model}...")
        with open(os.path.join(data_args.vqa_question_dir, f"{perspective}.jsonl"), "r") as f:
            data = [json.loads(line) for line in f]
        truths = [item["choices"].index(item["answer"]) for item in data]

        answers = batched_evaluate(data)

        output_dir = os.path.join(data_args.vqa_output_dir, vqa_args.eval_model)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{perspective}.txt"), "w") as f:
            f.write("\n".join(answers))
        logger.info(f"Evaluation results saved in {output_dir}/{perspective}.txt")

        # calculate the accuracy
        correct = sum(1 for pred, item in zip(answers, truths) if pred == item)
        accuracy = correct / len(data) * 100
        logger.info(f"{perspective} - Acc: {accuracy:.2f}, Correct: {correct}, Total: {len(data)}")


if __name__ == "__main__":
    main()

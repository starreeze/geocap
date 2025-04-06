# -*- coding: utf-8 -*-
# @Date    : 2024-12-09 19:07:47
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import importlib
import json
import os
import re
import sys
from typing import Any, TextIO

from tqdm import tqdm

from common.args import data_args, logger, run_args, vqa_args
from common.vllm.base import GenerateModelBase
from data.rule.utils import round_floats

Model = importlib.import_module(f"common.vllm.{vqa_args.eval_model.split('-')[0]}").GenerateModel


def batched_inference(model: GenerateModelBase, data: list[dict[str, Any]], f: TextIO) -> list[str]:
    batched_data = [
        data[i : i + vqa_args.eval_batchsize] for i in range(0, len(data), vqa_args.eval_batchsize)
    ]
    all_answer = []
    for batch in tqdm(batched_data):
        image_paths = [
            os.path.join(
                data_args.figure_dir,
                data_args.figure_name.format(prefix=data_args.figure_prefix, id=item["image_id"]),
            )
            for item in batch
        ]
        questions = [
            f"{item['question']}\n"
            + "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(item["choices"]))
            + f"\n{vqa_args.eval_inst}"
            for item in batch
        ]

        resps = model.generate(image_paths, questions)
        answers = [find_answer(resp, list(map(str, item["choices"]))) for resp, item in zip(resps, batch)]
        for resp, item, ans in zip(resps, batch, answers):
            f.write(json.dumps(item | {"response": resp, "pred": ans}) + "\n")
        all_answer.extend(answers)

    return all_answer


def find_answer(resp: str, choices: list[str]) -> str:
    words = re.findall(r"\b\w+\b", resp)  # Match full word A/B/C/D with word boundaries
    answer = "-"
    for word in words[::-1]:  # always find the last one
        if len(word) == 1 and word in "ABCD":
            answer = word
            break
    if answer != "-":
        return answer
    matches = [choice for choice in choices if choice in resp]
    if len(matches) == 1:
        answer = matches[0]
    return answer


def main():
    if run_args.end_pos != sys.maxsize or run_args.start_pos != 0:
        logger.warning(
            f"Evaluating only on {run_args.start_pos} - {run_args.end_pos} images; answers may not be aligned with the questions"
        )

    model: GenerateModelBase = Model(vqa_args.eval_model, max_new_tokens=10)
    scores: list[float] = []

    for perspective in vqa_args.perspectives:
        logger.info(f"Evaluating {perspective} on model {vqa_args.eval_model}...")
        with open(os.path.join(data_args.vqa_question_dir, f"{perspective}.jsonl"), "r") as f:
            data = [json.loads(line) for line in f]
        # only keep start_pos: end_pos image_ids
        data = list(filter(lambda x: run_args.start_pos <= x["image_id"] < run_args.end_pos, data))
        truths = [item["choices"].index(item["answer"]) for item in data]

        output_dir = os.path.join(data_args.vqa_output_dir, vqa_args.eval_model)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{perspective}.jsonl"), "w") as f:
            answers = batched_inference(model, data, f)
        logger.info(f"Evaluation results for {perspective} saved in {output_dir}/{perspective}.jsonl")

        # calculate the accuracy
        correct = sum(1 for pred, label in zip(answers, truths) if ord(pred) - ord("A") == label)
        accuracy = correct / len(data) * 100
        scores.append(accuracy)
        logger.info(f"{perspective} - Acc: {accuracy:.1f}, Correct: {correct}, Total: {len(data)}")

    with open(os.path.join(output_dir, f"scores.csv"), "w") as f:
        f.write(",".join(vqa_args.perspectives) + "\n")
        f.write(",".join(map(str, round_floats(scores, precision=1))) + "\n")
    logger.info(f"Evaluation results on model {vqa_args.eval_model} saved in {output_dir}/scores.csv")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# @Date    : 2024-12-03 11:18:24
# @Author  : Shangyu.Xing (starreeze@foxmail.com)
"construct VQA questions according to the captions and rules"

import json
import os

from common.args import data_args, vqa_args
from data.vqa.llm import LLMQAGenerator
from data.vqa.rule import RuleBasedQAGenerator


def main():
    os.makedirs(data_args.vqa_question_dir, exist_ok=True)

    with open(data_args.caption_path) as f:
        captions = [json.loads(line)["output"] for line in f]
    with open(data_args.rules_path) as f:
        rules = json.load(f)

    for perspective in vqa_args.perspectives:
        prompt_file = os.path.join(vqa_args.vqa_prompts_dir, f"{perspective}.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file, "r") as f:
                prompt = f.read()
            qa_generator = LLMQAGenerator(prompt)
            qa_pairs = qa_generator(captions)
        else:
            qa_generator = RuleBasedQAGenerator(rules)
            qa_pairs = qa_generator(perspective)

        with open(os.path.join(data_args.vqa_question_dir, f"{perspective}.jsonl"), "w") as f:
            f.write("\n".join([json.dumps(qa) for qa in qa_pairs]))


if __name__ == "__main__":
    main()

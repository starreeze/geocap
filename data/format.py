"convert generated dataset to MLLM trainable format"

from __future__ import annotations

import json
import os

from common.args import data_args


def to_llava():
    data: list[str] = open(data_args.caption_path, "r").readlines()
    target_data = []
    for i, d in enumerate(data):
        d = json.loads(d.strip())
        target_data.append(
            {
                "id": i,
                "image": data_args.figure_name.format(prefix=data_args.figure_prefix, id=i),
                "conversations": [
                    {"from": "human", "value": "<image>\n" + d["input"]},
                    {"from": "gpt", "value": d["output"]},
                ],
            }
        )
    os.makedirs(data_args.llava_data_dir, exist_ok=True)
    json.dump(target_data, open(data_args.llava_data_path, "w"), indent=2)


def to_llava_eval():
    "convert generated dataset to model_vqa_loader format"
    data: list[str] = open(data_args.caption_path, "r").readlines()
    target_data = []
    for i, d in enumerate(data):
        d = json.loads(d.strip())
        target_data.append(
            {
                "question_id": i,
                "text": d["input"],
                "image": data_args.figure_name.format(prefix=data_args.figure_prefix, id=i),
            }
        )
    os.makedirs(data_args.llava_data_dir, exist_ok=True)
    with open(data_args.llava_data_path, "w") as f:
        f.write("\n".join([json.dumps(d) for d in target_data]))


if __name__ == "__main__":
    to_llava()

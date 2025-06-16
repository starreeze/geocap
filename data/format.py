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


def to_internvl(input_path: str, output_path: str):
    file_type = input_path.split(".")[-1]
    with open(input_path, "r") as f:    
        if file_type == "jsonl":
            data = [json.loads(line) for line in f]
        elif file_type == "json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    assert output_path.endswith(".jsonl")
    with open(output_path, "w") as f:
        for i, d in enumerate(data):
            target_data = {
                "id": i,
                "image": d["image"],
                "conversations": [
                    {"from": "human", "value": "<image>\n" + d["input"]},
                    {"from": "gpt", "value": d["output"]},
                ],
            }
            f.write(json.dumps(target_data) + "\n")


def main():
    # to_llava()

    input_path = "dataset/stage2_captions/selected_captions_paraphrased.jsonl"
    output_path = input_path.replace(".jsonl", "_internvl.jsonl")
    to_internvl(input_path, output_path)


if __name__ == "__main__":
    main()

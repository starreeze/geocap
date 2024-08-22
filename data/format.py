"convert generated dataset to MLLM trainable format"

from __future__ import annotations
import os, json
from common.args import data_args


def to_llava():
    data: list[dict[str, str]] = json.load(open(data_args.captions_path, "r"))
    target_data = []
    for i, d in enumerate(data):
        target_data.append(
            {
                "id": i,
                "image": os.path.join(data_args.figure_dir, f"{i}.png"),
                "conversations": [
                    {"from": "human", "value": "<image>\n" + d["input"]},
                    {"from": "gpt", "value": d["output"]},
                ],
            }
        )
    json.dump(target_data, open(data_args.llava_data_path, "w"), indent=2)


if __name__ == "__main__":
    to_llava()

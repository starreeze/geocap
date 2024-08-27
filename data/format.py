"convert generated dataset to MLLM trainable format"

from __future__ import annotations

import json
import os

from common.args import data_args


def to_llava():
    # data: list[dict[str, str]] = json.load(open(data_args.captions_path, "r"))
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
    os.makedirs(os.path.dirname(data_args.llava_data_path), exist_ok=True)
    json.dump(target_data, open(data_args.llava_data_path, "w"), indent=2)


if __name__ == "__main__":
    to_llava()

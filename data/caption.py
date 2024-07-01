"construct descriptions according to generated rules"
import os, json
from typing import Any
from common.args import data_args


def caption(rules: list[dict[str, Any]], index: int) -> str:
    # TODO generate captions. Use whatever technique you like, e.g., demonstrations, cot, ...
    return ""


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
    captions = []
    for i, sample in enumerate(samples):
        captions.append(caption(sample, i))
    with open(data_args.captions_path, "w") as f:
        json.dump(captions, f)


if __name__ == "__main__":
    main()

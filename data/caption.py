"construct descriptions according to generated rules"
import os, json
from typing import Any
from common.args import data_args


def caption(rules: list[dict[str, Any]]) -> dict[str, str]:
    # TODO generate captions. Use whatever technique you like, e.g., demonstrations, cot, ...
    # TODO also consider how to handle distance:
    #      It should be mentioned in input prompt. Try scaling to produce different distances.
    # NOTE see common/llm.py for the use of open-source LLMs.

    return {"input": "", "output": ""}


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
    captions: list[dict[str, str]] = []
    for sample in samples:
        captions.append(caption(sample))
    with open(data_args.captions_path, "w") as f:
        json.dump(captions, f)


if __name__ == "__main__":
    main()

import json
import random

import jsonlines
from tqdm import tqdm

# from common.llm import generator_mapping, model_path_mapping
# from common.llm import APIGenerator  # sry but now we do this
from common.llm import APIGenerator


def sample_few_shot(training_set: list[dict], num_fewshot: int = 2) -> list[dict]:
    demonstrations = []
    sampled_examples = random.sample(training_set, num_fewshot)
    for i, example in enumerate(sampled_examples):
        demonstration = {
            "input": f"Example{i+1}:\nUser: {example['input']}",
            "image": example["image"],
            "output": f"Assistant: {example['output']}",
        }
        demonstrations.append(demonstration)
    return demonstrations


def main():
    with open("eval/prompts/gpt4o_4shot.json", "r") as f:
        prompts = json.load(f)
    sys_prompt = prompts[0]["system"]

    with open("dataset/no_vis_tools/add_default_value_train.jsonl", "r") as f:
        training_set = [json.loads(line) for line in f]

    generator = APIGenerator("gpt-4o-2024-11-20", sys_prompt=sys_prompt, temperature=1.0)

    with open("dataset/testset_no_vis_tools.jsonl", "r") as f:
        with jsonlines.open(
            "eval_data/outputs/gpt4o_4shot_sample_add_default_3.jsonl", mode="a", flush=True
        ) as writer:
            lines = list(f)
            pbar = tqdm(lines, desc="Generating responses")
            for line in pbar:
                data = json.loads(line)

                demonstrations = sample_few_shot(training_set, num_fewshot=4)
                # Format few shot prompt
                user_prompt = []
                for demonstration in demonstrations:
                    for key, value in demonstration.items():
                        if key == "input" or key == "output":
                            user_prompt.append(("text", value))
                        elif key == "image":
                            user_prompt.append(("image", f"dataset/common/images/{value}"))

                inputs = user_prompt + [
                    ("text", f"Your turn:\n{data['input']}"),
                    ("image", f"dataset/common/images/{data['image']}"),
                    ("text", "Assistant:"),
                ]
                resp = generator.get_one_response(inputs)
                jsonls = {"image": data["image"], "question": data["input"], "response": resp}
                # print(jsonls)
                writer.write(jsonls)


if __name__ == "__main__":
    main()

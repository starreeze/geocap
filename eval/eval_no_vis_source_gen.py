import json

import jsonlines
from tqdm import tqdm

# from common.llm import generator_mapping, model_path_mapping
# from common.llm import APIGenerator  # sry but now we do this
from common.llm import APIGenerator


def main():
    with open("eval/prompts/gpt4o_4shot.json", "r") as f:
        prompts = json.load(f)
    sys_prompt = prompts[0]["system"]
    demonstrations = prompts[1:]

    user_prompt = []
    for demonstration in demonstrations:
        for key, value in demonstration.items():
            if key == "input" or key == "output":
                user_prompt.append(("text", value))
            elif key == "image":
                user_prompt.append(("image", f"dataset/common/images/{value}"))

    generator = APIGenerator("gpt-4o-2024-11-20", sys_prompt=sys_prompt)

    with open("dataset/testset_no_vis_tools.jsonl", "r") as f:
        with jsonlines.open("eval_data/outputs/gpt4o_few_shot.jsonl", mode="a", flush=True) as writer:
            lines = list(f)
            pbar = tqdm(lines, desc="Generating responses")
            for line in pbar:
                data = json.loads(line)
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

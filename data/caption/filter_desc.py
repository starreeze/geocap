import json
import os

from tqdm import tqdm

from common.args import caption_args
from common.llm import APIGenerator
from iterwrap import iterate_wrapper


class DescFilter:
    def __init__(self) -> None:
        """Initialize the LLM generator for description filtering"""
        # Initialize prompt
        sys_prompt = "You are an expert data cleaning assistant."
        with open(caption_args.desc_filter_prompt_dir, "r") as f:
            self.user_prompt = f.read()

        # Initialize llm
        model_name, model_id = caption_args.caption_llm.split("-", 1)
        self.llm_generator = APIGenerator(model_id, max_tokens=8192, temperature=1.0, sys_prompt=sys_prompt)

    def _generate_prompt(self, text: str) -> list[dict[str, str]]:
        """Generate the prompt for description filtering"""
        user_prompt = self.user_prompt.replace("{text}", text)
        messages = [{"role": "user", "content": user_prompt}]
        return messages

    def __call__(self, text: str) -> str | None:
        """Filter a single text input"""
        messages = self._generate_prompt(text)
        response = self.llm_generator.get_one_response(messages)
        return response


def main():
    desc_filter = DescFilter()
    # Read original fossil captions
    with open("dataset/common/selected_data.json", "r") as f:
        data_dict = json.load(f)

    filtered_data_dict = data_dict
    fos_name_list = []
    desc_list = []
    for fos_name, data in data_dict.items():
        desc = data["desc"]
        fos_name_list.append(fos_name)
        desc_list.append(desc)

    filtered_desc_list = iterate_wrapper(desc_filter, desc_list)
    for fos_name, desc in zip(fos_name_list, filtered_desc_list):
        filtered_data_dict[fos_name]["desc"] = desc

    output_path = os.path.join("dataset/common/filtered_data.json")
    with open(output_path, "w") as f:
        json.dump(filtered_data_dict, f, indent=2)


if __name__ == "__main__":
    main()

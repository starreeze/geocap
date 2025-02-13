import os
import json

from tqdm import tqdm

from common.args import caption_args, data_args
from common.llm import generator_mapping, model_path_mapping


class DescFilter:
    def __init__(self) -> None:
        self.loaded_llm = False

    def load_llm_generator(self):
        """Initialize the LLM generator for description filtering"""
        assert not self.loaded_llm
        # Initialize llm
        model_name, model_id = caption_args.caption_llm.split("-", 1)
        model_path = model_path_mapping[model_name].format(model_id)
        self.llm_generator = generator_mapping[model_name](model_path, max_tokens=8192, temperature=1.0)
        self.model_name = model_name
        self.loaded_llm = True

        # Initialize prompt
        self.sys_prompt = "You are a helpful assistant."
        with open(caption_args.desc_filter_prompt_dir, "r") as f:
            self.user_prompt = f.read()

    def _generate_prompt(self, texts: list[str]) -> list[list[dict[str, str]]]:
        """Generate the prompt for description filtering"""
        user_prompts = [f"{self.user_prompt.replace('{text}', text)}" for text in texts]
        if "api" not in self.model_name:
            messages = [
                [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": user_prompt}]
                for user_prompt in user_prompts
            ]
        else:
            messages = [[{"role": "user", "content": user_prompt}] for user_prompt in user_prompts]
        return messages

    def __call__(self, texts: list[str]) -> list[str]:
        """Filter a single text input"""
        if not self.loaded_llm:
            self.load_llm_generator()

        messages = self._generate_prompt(texts)
        batch_size = 1 if "api" in self.model_name else caption_args.caption_batchsize
        responses = self.llm_generator(messages, batch_size)

        outputs = []
        total_batches = (len(messages) + batch_size - 1) // batch_size
        for batch in tqdm(responses, total=total_batches, desc="Filtering"):
            outputs.extend(batch)

        return outputs


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

    filtered_desc_list = desc_filter(desc_list)
    for fos_name, desc in zip(fos_name_list, filtered_desc_list):
        filtered_data_dict[fos_name]["desc"] = desc

    output_path = os.path.join("dataset/common/filtered_data.json")
    with open(output_path, "w") as f:
        json.dump(filtered_data_dict, f, indent=2)


if __name__ == "__main__":
    main()

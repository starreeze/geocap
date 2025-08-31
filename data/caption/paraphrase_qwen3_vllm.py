import json
import sys

from vllm import LLM, SamplingParams

from common.args import caption_args, data_args, run_args


class Paraphraser:
    def __init__(self, model_path: str) -> None:
        """Initialize the LLM generator for paraphrasing"""
        # Initialize llm
        self.model = LLM(
            model=model_path, max_model_len=8192, tensor_parallel_size=2, gpu_memory_utilization=0.8
        )
        self.sampling_params = SamplingParams(temperature=0.7, max_tokens=1024, top_p=0.8, top_k=20)

        # Initialize prompt
        self.sys_prompt = "You are a helpful assistant skilled in text paraphrasing."
        with open(caption_args.paraphrase_prompt_dir, "r") as f:
            self.user_prompt = f.read()

    def _generate_paraphrase_prompt(self, texts: list[str]) -> list[list[dict[str, str]]]:
        """Generate the prompt for paraphrasing"""
        user_prompts = [f"{self.user_prompt.replace('{text}', text)}" for text in texts]
        messages = [
            [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": user_prompt}]
            for user_prompt in user_prompts
        ]
        return messages

    def __call__(self, texts: list[str]) -> list[str]:
        """Paraphrase a single text input"""
        # messages = self._generate_paraphrase_prompt(texts)
        outputs = self.model.generate(texts, self.sampling_params)
        paraphrased_outputs = []
        for output in outputs:
            generated_text = output.outputs[0].text
            paraphrased_outputs.append(generated_text)
        return paraphrased_outputs


def main():
    model_path = "/home/nfs06/model/Qwen3-30B-A3B-Instruct-2507"
    paraphraser = Paraphraser(model_path)
    # Read original captions
    with open(data_args.caption_path, "r") as f:
        file_ext = data_args.caption_path.split(".")[-1]
        if file_ext == "jsonl":
            captions = [json.loads(line) for line in f]
        elif file_ext == "json":
            captions = json.load(f)
        else:
            raise ValueError(f"Unsupported file extension: {data_args.caption_path}")

    output_path = data_args.caption_path.replace(f".{file_ext}", "_paraphrased.jsonl")
    if run_args.end_pos != sys.maxsize:
        captions = captions[run_args.start_pos : run_args.end_pos]
        output_path = output_path.replace(".jsonl", f"_{run_args.start_pos}_{run_args.end_pos}.jsonl")

    # Extract and paraphrase outputs
    original_outputs = [caption["output"] for caption in captions]

    # Paraphrase captions
    paraphrased_outputs = paraphraser(original_outputs)

    # Replace original outputs with paraphrased ones
    caption_pairs = list(zip(captions, paraphrased_outputs))
    for caption, paraphrased_output in caption_pairs:
        caption["output"] = paraphrased_output

    with open(output_path, "w") as f:
        for caption in captions:
            f.write(json.dumps(caption) + "\n")


if __name__ == "__main__":
    main()

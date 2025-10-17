import json
import sys

from tqdm import tqdm
from transformers import AutoTokenizer
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Initialize prompt
        self.sys_prompt = "You are a helpful assistant skilled in text paraphrasing."
        with open(caption_args.paraphrase_prompt_dir, "r") as f:
            self.user_prompt = f.read()

    def generate_messages(self, texts: list[str]) -> list[list[dict[str, str]]]:
        """Generate the prompt for paraphrasing"""
        user_prompts = [f"{self.user_prompt.replace('{text}', text)}" for text in texts]
        messages = [
            [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": user_prompt}]
            for user_prompt in user_prompts
        ]
        return messages

    def __call__(self, data_batch: list[str]) -> list[str]:
        """Paraphrase a single text input"""
        messages_list = self.generate_messages(data_batch)
        text = self.tokenizer.apply_chat_template(messages_list, tokenize=False, add_generation_prompt=True)
        outputs = self.model.generate(text, self.sampling_params)
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
    bs = caption_args.caption_batchsize
    total_steps = len(captions) // bs

    with open(output_path, "a") as f:
        for i in tqdm(range(0, len(captions), bs), total=total_steps, desc="Paraphrasing", position=2):
            data_batch = original_outputs[i : i + bs]
            paraphrased_outputs = paraphraser(data_batch)
            for caption, paraphrased_output in zip(captions, paraphrased_outputs):
                caption["output"] = paraphrased_output
                f.write(json.dumps(caption) + "\n")
                f.flush()


if __name__ == "__main__":
    main()

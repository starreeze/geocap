import argparse
import json

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

demo_prompt = """
Please read the following examples. Then extract the answer from the model response and type it at the end of the prompt.
### Example 1 ###
Question: What percentage of female students achieved a C/4 grade or higher in the United Kingdom in 2020?
Model response: In the image, the percentage of female students achieving a C/4 grade or higher in the United Kingdom in 2020 is shown as 70%.
Extracted answer: 70

### Example 2 ###
Question: Which late night host had the highest favorability ratings??
Model response: According to the chart, the late night host with the highest favorability ratings was Jimmy Fallon.
Extracted answer: Jimmy Fallon
"""


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def generate(model_name, prompt, res):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ans = []
    for query, response in tqdm(zip(prompt, res), total=len(prompt)):
        full_prompt = create_test_prompt(demo_prompt, query, response)
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": full_prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ans.append(response)
    return ans


def parse_args():
    parser = argparse.ArgumentParser(description="Process input paths and generate answers.")
    parser.add_argument(
        "--response_file",
        type=str,
        required=False,
        default="/home/nfs05/xingsy/wzt/geocap/ChartQA/ChartQADataset/test/results/answer.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="/home/nfs05/xingsy/wzt/geocap/model/Qwen2.5-14B-Instruct",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        default="/home/nfs05/xingsy/wzt/geocap/ChartQA/ChartQADataset/test/results/ouput.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.response_file, "r") as f:
        data = [json.loads(line) for line in f]
    prompt = []
    res = []
    for i in range(len(data)):
        query = data[i]["question"]
        response = data[i]["response"]
        prompt.append(query)
        res.append(response)
    extractions = generate(args.model_path, prompt, res)
    extractions_list = []
    for i, response in enumerate(extractions):
        extractions_list.append(
            {
                "answer": data[i].get("answer", i),
                "extraction": extractions[i],
                "response": data[i].get("response", i),
            }
        )
    with open(args.output_file, "w") as f:
        for extraction in extractions_list:
            f.write(json.dumps(extraction) + "\n")
    print(f"extractions saved to {args.output_file}")

import argparse
import json
from collections import OrderedDict

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14
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
    for query, response in tqdm(zip(prompt, res), total=len(res)):
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
        "--query_file",
        type=str,
        required=False,
        default="MathVista/data/query.json",
        help="Path to the query JSON file (default: MathVista/data/query.json).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=False,
        default="MathVista/data/testmini.json",
        help="Path to the test JSON file (default: MathVista/data/testmini.json).",
    )
    parser.add_argument(
        "--response_file",
        type=str,
        required=False,
        default="/home/nfs06/xingsy/wzt/geocap/MathVista/data/answer.json",
        help="Path to the response JSON file (default: /home/nfs06/xingsy/wzt/geocap/MathVista/data/output.json).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="/home/nfs02/model/Qwen_Qwen2.5-7B-Instruct",
        help="Path to the model (default: /home/nfs02/model/Qwen_Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        default="MathVista/data/updated_data.json",
        help="Path to save the updated data JSON file (default: MathVista/data/updated_data.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.query_file, "r") as f:
        data1 = json.load(f)

    with open(args.test_file, "r") as f:
        data2 = json.load(f)

    with open(args.response_file, "r") as f:
        data3 = json.load(f)

    answers = OrderedDict()
    prompt = []
    res = []

    for i in range(1, 1001):
        pid = str(i)
        question_type = data2[pid]["question_type"]
        answer_type = data2[pid]["answer_type"]
        choices = data2[pid]["choices"]
        query = data1[pid]
        response = data3[i - 1]

        if response == "":
            answers[pid] = {"pid": pid, "answer": ""}
            continue

        if question_type == "multi_choice":
            answers[pid] = {"pid": pid, "answer": response}
            continue

        if answer_type == "integer":
            try:
                extraction = int(response)
                answers[pid] = {"pid": pid, "answer": extraction}
                continue
            except ValueError:
                pass

        if answer_type == "float":
            try:
                extraction = float(response)
                answers[pid] = {"pid": pid, "answer": extraction}
                continue
            except ValueError:
                pass
        answers[pid] = {"pid": pid, "answer": "llavageo"}
        prompt.append(query)
        res.append(response)

    extractions = generate(args.model_path, prompt, res)
    idx = 0
    for pid in answers.keys():
        if answers[pid]["answer"] == "llavageo":
            print(pid)
            print(idx)
            answers[pid] = {"pid": pid, "answer": extractions[idx]}
            idx += 1

    for pid, answer in answers.items():
        if pid in data2:
            data2[pid]["extraction"] = answer["answer"]

    with open(args.output_file, "w") as f:
        json.dump(data2, f, indent=4)

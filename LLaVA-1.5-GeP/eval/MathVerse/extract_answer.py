import argparse
import copy
import json
import os
from collections import defaultdict

from prompts import demo_prompt_extract
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *


def verify_extraction(extraction):
    extraction = extraction.strip()
    if not extraction:
        return False
    return True


def create_test_prompt(demo_prompt, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"Model response: '{response}'\nExtracted Answer: "
    full_prompt = f"{demo_prompt}\n\n{test_prompt}"
    return full_prompt


def generate(model, tokenizer, demo_prompt, responses):
    ans = []
    for response in tqdm(responses, desc="Generating answers"):
        full_prompt = create_test_prompt(demo_prompt, response)
        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": full_prompt},
        ]
        text = tokenizer.apply_chat_template(  # type: ignore
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ans.append(response)
    return ans


def extract_answers(responses, model, tokenizer):
    try:
        extractions = generate(model, tokenizer, demo_prompt_extract, responses)
        return extractions
    except Exception as e:
        print(e)
        print("Error in extracting answers")
    return []


def trunk_response(response, trunk_length):
    if trunk_length <= 0:
        return response
    else:
        return_res = " ".join(response.split(" ")[-trunk_length:])
        return return_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_output_file", type=str, default="MathVerse/data/output.json")
    parser.add_argument("--save_file", type=str, default="MathVerse/data/answer.json")
    parser.add_argument("--trunk_response", type=int, default=-1, help="trunk response to the last n words")
    parser.add_argument(
        "--model_path", type=str, help="Path to the model", default="./model/Qwen2.5-14B-Instruct"
    )
    args = parser.parse_args()

    # read results
    result_file = args.model_output_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    save_results = []

    score_dict = defaultdict(lambda: defaultdict(list))
    score_dict_record = defaultdict(list)
    score_version_dict = defaultdict(list)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    batch_responses = []
    batch_insts = []
    for i, inst in enumerate(results):
        save_inst = save_results[i] if i < len(save_results) else copy.deepcopy(inst)
        if "model_answer" in save_inst:
            response = save_inst["model_answer"]
        else:
            response = ""
            print(save_inst)
            print("######### NO MODEL ANSWER ###########")
        response = trunk_response(response, args.trunk_response)

        batch_responses.append(response)
        batch_insts.append(save_inst)
    extractions = extract_answers(batch_responses, model, tokenizer)
    for j, save_inst in enumerate(batch_insts):
        save_inst["extraction"] = extractions[j].replace("Extracted Answer: ", "").strip()
        save_results.append(save_inst)
    print(f"Saving results to {args.save_file}...")
    with open(args.save_file, "w") as f:
        json.dump(save_results, f, indent=4)
    print(f"Results saved.")

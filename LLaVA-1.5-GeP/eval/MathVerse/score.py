import argparse
import copy
import json
import os
import re
from collections import defaultdict

from prompts import demo_prompt_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import *


# load demo prompt
def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction is None:
        return False
    return True


def create_test_prompt(demo_prompt, inst):
    demo_prompt = demo_prompt.strip()
    full_prompt = demo_prompt.format(
        question=inst["question"], gt=inst["answer"], extraction=inst["extraction"]
    )
    return full_prompt


def match_answer(insts, model, tokenizer, quick_match):
    # quick match
    res = []
    for inst in tqdm(insts):
        if quick_match:
            if inst["answer"] == inst["extraction"]:
                response = "1"
            else:
                response = "0"
            res.append(response)
            continue
        try:
            # Create test prompt for matching
            full_prompt = create_test_prompt(demo_prompt_score, inst)

            # Use Qwen-2.7B model for answer extraction
            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": full_prompt},
            ]

            # Tokenize and generate response
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
            response = response.replace("Judgement:", "").strip()
            for c in [0, 1]:
                if response.startswith(str(c)) or response.startswith(f"{c}\n\n"):
                    extracted_number = str(c)
                    response = extracted_number
            res.append(response)
        except Exception as e:
            print(e)
            print("Error in matching answer")
            response = ""
            res.append(response)
    return res


def trunk_response(response, trunk_length):
    if trunk_length <= 0:
        return response
    else:
        return_res = " ".join(response.split(" ")[-trunk_length:])
        return return_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # input
    parser.add_argument("--answer_extraction_file", type=str, default="MathVerse/data/answer.json")
    parser.add_argument("--save_file", type=str, default="MathVerse/data/answer1.json")
    # output
    parser.add_argument("--save_every", type=int, default=10, help="save every n problems")
    parser.add_argument("--cache", action="store_true", help="cache results")
    parser.add_argument("--trunk_response", type=int, default=-1, help="trunk response to the last n words")
    # args
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the Qwen-2.7B model",
        default="/home/nfs05/xingsy/wzt/geocap/model/Qwen2.5-14B-Instruct",
    )

    args = parser.parse_args()

    # read results
    result_file = args.answer_extraction_file
    print(f"Reading {result_file}...")
    results = read_json(result_file)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)

    save_results = []

    score_dict = defaultdict(lambda: defaultdict(list))
    score_version_dict = defaultdict(list)

    # Initialize the model and tokenizer once
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    batch_insts = []

    # tqdm, enumerate results
    for i, inst in enumerate(results):
        save_inst = save_results[i] if i < len(save_results) else copy.deepcopy(inst)
        if args.cache and "judgement" in save_inst:
            pass
        else:
            batch_insts.append(save_inst)

    # Match answers in batch
    judgements = match_answer(batch_insts, model, tokenizer, False)
    judgements_list = [
        {"instance_id": save_inst["sample_index"], "judgement": judgements[j]}
        for j, save_inst in enumerate(batch_insts)
    ]
    save_json(judgements_list, "MathVerse/data/judgements.json")
    # Update the results with judgements
    unmatck_ids = []
    for j, save_inst in enumerate(batch_insts):
        if judgements[j].strip() not in ["0", "1"]:
            judgements[j] = "-1"
            unmatck_ids.append(j)
        save_inst["judgement"] = int(judgements[j])
        save_results.append(save_inst)
        score_dict[save_inst["metadata"]["subject"]][save_inst["metadata"]["subfield"]].append(
            save_inst["judgement"]
        )
        score_version_dict[save_inst["problem_version"]].append(save_inst["judgement"])

    save_json(save_results, args.save_file)
    print(f"Results saved.")

    # subject level acc
    total_cnt, right_cnt = 0, 0
    for subject in score_dict:
        subject_total_cnt, subject_right_cnt = 0, 0
        for subfield in score_dict[subject]:
            subfield_total_cnt = len(score_dict[subject][subfield])
            subfield_right_cnt = len([inst for inst in score_dict[subject][subfield] if inst == 1])
            subject_total_cnt += subfield_total_cnt
            subject_right_cnt += subfield_right_cnt
            print(f"{subject}-{subfield} Acc: {(subfield_right_cnt/subfield_total_cnt):.3f}")
        print(f"{subject} Acc: {(subject_right_cnt/subject_total_cnt):.3f}")
        total_cnt += subject_total_cnt
        right_cnt += subject_right_cnt
    print(f"Total Acc: {(right_cnt/total_cnt):.3f}")

    # version level acc
    total_cnt, right_cnt = 0, 0
    for version in score_version_dict:
        version_total_cnt = len(score_version_dict[version])
        version_right_cnt = len([inst for inst in score_version_dict[version] if inst == 1])
        total_cnt += version_total_cnt
        right_cnt += version_right_cnt
        print(f"{version} Acc: {(version_right_cnt/version_total_cnt):.3f}")
        print(version_total_cnt)

    print(f"Acc: {(right_cnt/total_cnt):.3f}")
    print(unmatck_ids)

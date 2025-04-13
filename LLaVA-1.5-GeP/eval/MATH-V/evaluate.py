import argparse
import json
import os
import re

from tqdm import tqdm
from utils import find_math_answer, is_equal, is_number, load_jsonl, save_jsonl

id_raw = {example["id"]: example for example in load_jsonl("./MATH-V/data/test.jsonl")}


def evaluate(answer_file, regen_answer=True):
    lines = load_jsonl(answer_file)
    for line in tqdm(lines, desc="gen_correct"):
        raw_exampe = id_raw[line["id"]]

        gt_answer = str(raw_exampe["answer"])
        if len(raw_exampe["options"]) > 0:
            gt_answer_value = raw_exampe["options"][ord(gt_answer) - ord("A")]
        else:
            gt_answer_value = ""

        if "model_answer" not in line or regen_answer:
            model_answer = line["response"].strip()
            for c in "ABCDE":
                if re.search(rf"\b{c}\b", model_answer):
                    model_answer = c
                    break
            if is_number(model_answer.split("is ")[-1].rstrip(".")):
                model_answer = model_answer.split("is ")[-1].rstrip(".")
            if "oxed{" not in model_answer:
                for flag in [
                    "the final answer is",
                    "the answer is",
                    "the correct answer is",
                    "the answer should be",
                ]:
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split("\n")[0].split(". ")[0]
                    flag = flag.replace("the", "The")
                    raw_model_answer = model_answer
                    model_answer = model_answer.split(flag)[-1].strip()
                    if flag in raw_model_answer:
                        model_answer = model_answer.split("\n")[0].split(". ")[0]
            elif model_answer.count("oxed{") > 1:
                model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]

            model_answer = (
                find_math_answer(model_answer)
                .replace("(a)", "a")
                .replace("(b)", "b")
                .replace("(c)", "c")
                .replace("(d)", "d")
                .replace("(e)", "e")
                .replace("{a}", "a")
                .replace("{b}", "b")
                .replace("{c}", "c")
                .replace("{d}", "d")
                .replace("{e}", "e")
                .rstrip(".")
                .lstrip(":")
                .strip()
            )
            line["model_answer"] = model_answer
        else:
            model_answer = line["model_answer"]
        line["correct"] = is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer)
    save_jsonl(answer_file, lines, t_stamp=False)


def math_level_subject_acc(answer_file):
    print(answer_file)
    lines = load_jsonl(answer_file)

    results_dict = {}
    for line in tqdm(lines, desc="math_level_subject_acc"):
        correct = line["correct"]
        raw_exampe = id_raw[line["id"]]
        subject = raw_exampe["subject"]
        level = raw_exampe["level"]
        for key in [
            "-all",
            f"-level{level}",
            f"{subject}",
            f"{subject}_level{level}",
            f"-level{level}_{subject}",
        ]:
            if key not in results_dict:
                results_dict[key] = [0, 0]
            results_dict[key][0] += 1 if correct else 0
            results_dict[key][1] += 1

    for key in results_dict.keys():
        if results_dict[key][1] == 0:
            results_dict[key] = f"{results_dict[key][0]}/{results_dict[key][1]}=0"
        else:
            results_dict[key] = (
                f"{results_dict[key][0]}/{results_dict[key][1]}={round(results_dict[key][0]/ max(results_dict[key][1], 1)*100, 2)}%"
            )

    results_dict = {key: results_dict[key] for key in sorted(results_dict.keys())}
    print(os.path.basename(answer_file), ":\t", results_dict["-all"])
    json.dump(
        results_dict, open(answer_file.replace(".json", "_result.json"), "w"), indent=4, ensure_ascii=False
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate model from images and prompts.")
    parser.add_argument(
        "--response_file",
        type=str,
        help="Path to save the output JSON response file.",
        default="/path/to/response.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.response_file)
    math_level_subject_acc(args.response_file)

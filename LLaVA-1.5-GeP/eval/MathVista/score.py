import argparse
import json
import logging
import os
import re

import pandas as pd

from Levenshtein import distance
from rich.logging import RichHandler
from tqdm import tqdm


def get_most_similar(prediction, choices):
    distances = [distance(prediction, choice) for choice in choices]
    ind = distances.index(min(distances))
    return choices[ind]


def normalize_extracted_answer(
    extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=True
):
    """
    Normalize the extracted answer to match the answer type
    """
    if question_type == "multi_choice":
        # make sure the extraction is a string
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        # if the extraction is empty, return None
        if ignore_empty_extractions and not extraction:
            return None

        # extract "A" from "(A) text"
        letter = re.findall(r"\(([a-zA-Z])\)", extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord("A") + i) for i in range(len(choices))]

        # if model output a character, use it as index of available choices
        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            # select the most similar option
            normalized_extraction = get_most_similar(extraction, choices)
        assert normalized_extraction in choices

    elif answer_type == "integer":
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == "float":
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None

    elif answer_type == "list":
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction


def safe_equal(prediction, answer):
    """
    Check if the prediction is equal to the answer, even if they are of different types
    """
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        logging.info(e)
        return False


def get_acc_with_contion(res_pd, key, value):
    if key == "skills":
        # if value in res_pd[key]:
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]

    correct_pd = total_pd[total_pd["true_false"] == True]
    acc = len(correct_pd) / len(total_pd)

    return len(correct_pd), len(total_pd), acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth_file", type=str, default="MathVista/data/testmini.json")
    parser.add_argument("--output_file", type=str, default="MathVista/data/updated_data.json")
    parser.add_argument("--score_file", type=str, default="MathVista/data/llava-v1.5-7b_metrics.json")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(args.groundtruth_file, "r") as f:
        data_list = json.load(f)
    with open(args.output_file, "r") as f:
        results = json.load(f)

    test_pids = list(results.keys())
    update_json_flag = False
    for pid in tqdm(test_pids):
        problem = results[pid]
        choices = problem["choices"]
        question_type = problem["question_type"]
        answer_type = problem["answer_type"]
        precision = problem["precision"]
        extraction = problem["extraction"]

        if "answer" in problem:
            answer = problem["answer"]
        else:
            answer = data_list[pid]["answer"]
            problem["answer"] = answer

        # normalize the extracted answer to match the answer type
        prediction = normalize_extracted_answer(
            extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=True
        )

        # verify the prediction is true or false
        true_false = safe_equal(prediction, answer)

        # update the problem
        if "true_false" not in problem:
            update_json_flag = True

        elif true_false != problem["true_false"]:
            update_json_flag = True

        if "prediction" not in problem:
            update_json_flag = True

        elif prediction != problem["prediction"]:
            update_json_flag = True

        problem["prediction"] = prediction
        problem["true_false"] = true_false

    # save the updated json
    if update_json_flag:
        logging.info("Updating input file with predictions and true_false...")
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=4)

    logging.info("Calculate the average accuracy")
    total = len(test_pids)
    correct = 0
    for pid in tqdm(test_pids):
        if results[pid]["true_false"]:
            correct += 1

    accuracy = correct / total
    print(accuracy)
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    # [3] Calculate the fine-grained accuracy scores
    # merge the 'metadata' attribute into the data
    for pid in results:
        results[pid].update(results[pid].pop("metadata"))

    # convert the data to a pandas DataFrame
    results_df = pd.DataFrame(results).T

    # asign the target keys for evaluation
    target_keys = [
        "question_type",
        "answer_type",
        "language",
        "source",
        "category",
        "task",
        "context",
        "grade",
        "skills",
    ]

    for key in target_keys:
        # get the unique values of the key
        if key == "skills":
            # the value is a list
            values = []
            for i in range(len(results_df)):
                values += results_df[key][i]
            values = list(set(values))
        else:
            values = results_df[key].unique()

        # calculate the accuracy for each value
        scores[key] = {}
        for value in values:
            correct, total, acc = get_acc_with_contion(results_df, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}

        # sort the scores by accuracy
        scores[key] = dict(
            sorted(scores[key].items(), key=lambda item: float(item[1]["accuracy"]), reverse=True)
        )

    metrics_str = get_full_metrics_str(scores)
    logging.info(metrics_str)
    with open(args.score_file, "w") as f:
        json.dump(scores, f, indent=4)
    logging.info("MathVista: Calculating Scores - Finish")


def get_full_metrics_str(metrics_dict) -> str:
    divider = "=" * 40

    avg_accuracy = metrics_dict["average"]["accuracy"]
    avg_correct = metrics_dict["average"]["correct"]
    avg_total = metrics_dict["average"]["total"]

    metrics_str = f"""
{f"Correct: {avg_correct}/{avg_total} - Accuracy: {avg_accuracy * 100:.2f}%"}
{divider}
""".lstrip()

    for key, item in metrics_dict.items():
        if key == "average":
            continue

        formatted_item_dict = {}
        for sub_key, sub_item in item.items():
            acc = sub_item["accuracy"]
            correct = sub_item["correct"]
            total = sub_item["total"]
            values = [f"{acc * 100:.2f}%", f"({correct}/{total})"]

            formatted_item_dict[sub_key] = values

        category_df = pd.DataFrame(formatted_item_dict, index=["Accuracy", "Correct/Total"])

        metrics_str += f"""
{key}
{divider}
{category_df.T}
"""

    return metrics_str


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(rich_tracebacks=True, markup=False, show_path=False, omit_repeated_times=False)
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()

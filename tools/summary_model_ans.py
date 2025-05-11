import argparse
import json
import os

__curdir__ = os.path.dirname(__file__)
working_dir = os.path.join(__curdir__, ".analyse_results")
os.makedirs(working_dir, exist_ok=True)

perspectives = ["counting", "existing", "location", "reference", "relation", "size"]
std_ans_dict = {}


def collect_std_ans(questions_path):
    global std_ans_dict
    if os.path.exists(os.path.join(working_dir, "std_ans.json")):
        with open(os.path.join(working_dir, "std_ans.json"), "r") as f:
            std_ans_dict = {
                perspective: {tuple(k): v for k, v in sub_dict}
                for perspective, sub_dict in json.load(f).items()
            }
    else:
        std_ans_dict = {}

        for filename in os.listdir(questions_path):
            if filename.removesuffix(".jsonl") not in perspectives:
                continue
            perspective = filename.removesuffix(".jsonl")
            std_ans_dict[perspective] = {}
            with open(os.path.join(questions_path, filename), "r") as f:
                while line := f.readline():
                    item = json.loads(line)
                    std_ans_dict[perspective][item["image_id"], item["question_id"]] = "ABCD"[
                        item["choices"].index(item["answer"])
                    ]

        with open(os.path.join(working_dir, "std_ans.json"), "w") as f:
            json.dump(
                {
                    perspective: [(k, v) for k, v in sub_dict.items()]
                    for perspective, sub_dict in std_ans_dict.items()
                },
                f,
            )


def summary_model_ans(results_path):
    model_results = {}
    for model_name in os.listdir(results_path):
        model_results_path = os.path.join(results_path, model_name)
        _valid = []
        for perspective in perspectives:
            if os.path.exists(os.path.join(model_results_path, perspective + ".jsonl")):
                _valid.append(perspective)
        if _valid:
            model_results[model_name] = _valid

    model_cnt = len(model_results)
    header = ["__std__"] + list(model_results.keys())
    model_ans_dict = {
        perspective: {img_q_id: [ans] + ["-"] * model_cnt for img_q_id, ans in sub_dict.items()}
        for perspective, sub_dict in std_ans_dict.items()
    }

    for model_idx, model_name in enumerate(header[1:], start=1):
        model_results_path = os.path.join(results_path, model_name)
        for perspective in model_results[model_name]:
            with open(os.path.join(model_results_path, perspective + ".jsonl"), "r") as f:
                while line := f.readline():
                    item = json.loads(line)
                    model_ans_dict[perspective][item["image_id"], item["question_id"]][model_idx] = item[
                        "pred"
                    ]

    with open(os.path.join(working_dir, "model_ans.json"), "w") as f:
        json.dump(
            {".header": header}
            | {
                perspective: [(k, v) for k, v in sub_dict.items()]
                for perspective, sub_dict in model_ans_dict.items()
            },
            f,
        )


parser = argparse.ArgumentParser(
    description="Collect the standard answer and all models' evaluation results."
)
parser.add_argument("--questions_path", type=str, required=True, help="Path to questions directory")
parser.add_argument(
    "--results_path", type=str, required=True, help="Path to the directory of models' evaluation."
)
args = parser.parse_args()

collect_std_ans(args.questions_path)
summary_model_ans(args.results_path)

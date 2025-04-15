import argparse
import json


def process_data(file_path):
    count = 0
    with open(file_path, "r") as file:
        for line in file:
            data = json.loads(line.strip())
            answer = data["answer"]
            extraction = data["extraction"]
            try:
                answer_float = float(answer)
                extraction_number = float("".join(filter(str.isdigit, extraction)))
                if answer_float == 0:
                    if abs(answer_float - extraction_number) < 1e-5:
                        count += 1
                else:
                    relative_error = abs(answer_float - extraction_number) / answer_float
                    if relative_error <= 0.05:
                        count += 1
            except ValueError:
                if answer.strip().lower() in extraction.strip().lower():
                    count += 1
    print(f"Total acc: {count/1250}")


def parse_args():
    parser = argparse.ArgumentParser(description="Process input paths and generate answers.")
    parser.add_argument(
        "--input_file", type=str, required=False, default="ChartQA/ChartQADataset/test/results/ouput.json"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = args.input_file
    process_data(file_path)

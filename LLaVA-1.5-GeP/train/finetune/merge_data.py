import json
import random

random.seed(42)
file_paths = [
    "./playground/data/llava_v1_5_mix665k.json",
    "LLaVA-1.5-GeP/train/finetune/easy-counting.jsonl",
    "LLaVA-1.5-GeP/train/finetune/easy-existence.jsonl",
    "LLaVA-1.5-GeP/train/finetune/easy-reference.jsonl",
    "LLaVA-1.5-GeP/train/finetune/easy-size.jsonl",
    "LLaVA-1.5-GeP/train/finetune/easy-relation.jsonl",
    "LLaVA-1.5-GeP/train/finetune/easy-location.jsonl",
    "LLaVA-1.5-GeP/train/finetune/hard-counting.jsonl",
    "LLaVA-1.5-GeP/train/finetune/hard-existence.jsonl",
    "LLaVA-1.5-GeP/train/finetune/hard-reference.jsonl",
    "LLaVA-1.5-GeP/train/finetune/hard-size.jsonl",
    "LLaVA-1.5-GeP/train/finetune/hard-relation.jsonl",
    "LLaVA-1.5-GeP/train/finetune/hard-location.jsonl",
]

combined_data = []


with open(file_paths[0], "r") as f:
    data = json.load(f)
    combined_data.extend(data)

for file_path in file_paths[1:]:
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            combined_data.append(data)


random.shuffle(combined_data)


output_file = "LLaVA-1.5-GeP/train/finetune/gllava.json"
with open(output_file, "w") as f:
    json.dump(combined_data, f, indent=4)

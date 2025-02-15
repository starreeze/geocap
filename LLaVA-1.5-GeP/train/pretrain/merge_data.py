import json
import random


file_paths = [
    "./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json",
    "LLaVA-1.5-GeP/train/pretrain/easy_complete.jsonl",
    "LLaVA-1.5-GeP/train/pretrain/hard_complete.jsonl",
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


output_file = "LLaVA-1.5-GeP/train/pretrain/gllava.json"
with open(output_file, "w") as f:
    json.dump(combined_data, f, indent=4)
print(f"{output_file}")

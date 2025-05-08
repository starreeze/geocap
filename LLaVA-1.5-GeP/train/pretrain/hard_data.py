import json

input_files = [
    "/captions-hard/n0_011k.jsonl",
    "/captions-hard/n0_022k.jsonl",
    "/captions-hard/n0_033k.jsonl",
    "/captions-hard/n0_045k.jsonl",
    "/captions-hard/n0_056k.jsonl",
    "/captions-hard/n0_075k.jsonl",
    "/captions-hard/n0_093k.jsonl",
    "/captions-hard/n0_112k.jsonl",
    "/captions-hard/n0_131k.jsonl",
    "/captions-hard/n0_150k.jsonl",
]


complete_output_file = "LLaVA-1.5-GeP/train/pretrain/hard_complete.jsonl"


all_data = []
idx = 0


for input_file in input_files:
    with open(input_file, "r") as infile:
        for line in infile:
            data = json.loads(line.strip())

            converted_data = {
                "id": f"hard_{idx:08d}",
                "image": f"hard/hard_{idx:08d}.jpg",
                "conversations": [
                    {"from": "human", "value": data["input"] + "\n<image>"},
                    {"from": "gpt", "value": data["output"]},
                ],
            }
            all_data.append(converted_data)
            idx += 1


with open(complete_output_file, "w") as outfile:
    for item in all_data:
        outfile.write(json.dumps(item) + "\n")

print(f"{complete_output_file}")

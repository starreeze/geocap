import json

input_files = [
    "/captions-easy/n0_018k.jsonl",
    "/captions-easy/n0_037k.jsonl",
    "/captions-easy/n0_056k.jsonl",
    "/captions-easy/n0_075k.jsonl",
    "/captions-easy/n0_093k.jsonl",
    "/captions-easy/n0_112k.jsonl",
    "/captions-easy/n0_131k.jsonl",
    "/captions-easy/n0_150k.jsonl",
]

complete_output_file = "/LLaVA-1.5-GeP/train/pretrain/easy_complete.jsonl"

all_data = []
idx = 0


for input_file in input_files:
    with open(input_file, "r") as infile:
        for line in infile:
            data = json.loads(line.strip())
            converted_data = {
                "id": f"easy_{idx:08d}",
                "image": f"easy/easy_{idx:08d}.jpg",
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

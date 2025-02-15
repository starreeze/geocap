import json
import random
import os

random.seed(42)

input_files = {
    "/vqa-hard/counting.jsonl": 20000,
    "/vqa-hard/existence.jsonl": 20000,
    "/vqa-hard/reference.jsonl": 20000,
    "/vqa-hard/size.jsonl": 20000,
    "/vqa-hard/relation.jsonl": 20000,
    "/vqa-hard/location.jsonl": 20000,
}

remove_choices_categories = ["counting", "location"]

for input_file, sample_size in input_files.items():
    all_data = []

    category = os.path.basename(input_file).split(".")[0]

    with open(input_file, "r") as infile:
        lines = infile.readlines()
        num_samples = min(sample_size, len(lines))
        selected_data = random.sample(lines, num_samples)

        for line in selected_data:
            data = json.loads(line.strip())
            image_id = data["image_id"]
            question = data["question"]
            choices = data["choices"]
            answer = data["answer"]

            if category == "existence" and question.lower().startswith("is"):
                conversation_value = f"<image>\n{question}"
            elif category in remove_choices_categories:
                conversation_value = f"<image>\n{question}"
            else:
                conversation_value = f"<image>\n{question} Select one from the following options: {choices}"

            converted_data = {
                "id": f"hard_{image_id:08d}",
                "image": f"hard/hard_{image_id:08d}.jpg",
                "conversations": [
                    {"from": "human", "value": conversation_value},
                    {"from": "gpt", "value": str(answer)},
                ],
            }

            all_data.append(converted_data)

    output_file_name = f"hard-{os.path.basename(input_file)}"
    output_file = os.path.join("LLaVA-1.5-GeP/train/finetune", output_file_name)

    with open(output_file, "w") as outfile:
        for item in all_data:
            outfile.write(json.dumps(item) + "\n")

    print(f"Converted data written to: {output_file}")

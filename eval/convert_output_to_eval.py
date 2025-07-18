import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("generated_path", type=str, nargs="?", help="Path to the generated output file")
    parser.add_argument(
        "--reference_path",
        type=str,
        default="dataset/stage3/no_vis/add_default_value.jsonl",
        help="Path to the reference file",
    )
    args = parser.parse_args()

    generated_path = args.generated_path
    reference_path = args.reference_path

    with open(generated_path, "r", encoding="utf-8") as f:
        outputs = [json.loads(line) for line in f]

    with open(reference_path, "r", encoding="utf-8") as f:
        references = [json.loads(line) for line in f]

    # Create a dictionary to map image names to their references
    reference_dict = {}
    for ref in references:
        reference_dict[ref["image"]] = ref["output"]

    # Create the evaluation data structure
    eval_data = []
    for output in outputs:
        img_name = output["image"]
        if img_name in reference_dict:
            eval_item = {
                "img": img_name,
                "question": output["question"],
                "output": output["response"],
                "reference": reference_dict[img_name],
            }
            eval_data.append(eval_item)
        else:
            print(f"Warning: No reference found for image {img_name}")

    # Save the evaluation data to a JSON file
    output_file_name = generated_path.split("\\")[-1].replace(".jsonl", ".json")
    origin_file_path = f"eval_data/origin_files/{output_file_name}"
    with open(origin_file_path, "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"Evaluation data saved to {origin_file_path}")
    print(f"Total evaluation pairs: {len(eval_data)}")

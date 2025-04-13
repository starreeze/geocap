import argparse
import json
import os

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=False, default="llava-1.5-13b")
    parser.add_argument("--split", type=str, required=False, default="val")
    parser.add_argument("--mode", type=str, required=False, default="reasoning")
    parser.add_argument("--gen_prefix", type=str, default="gen-")
    args = parser.parse_args()
    args.input_file = f"CharXiv/data/{args.mode}_{args.split}.json"
    args.resp_file = f"CharXiv/results/{args.gen_prefix}{args.model_name}-{args.mode}_{args.split}.json"
    args.output_file = args.resp_file.replace(args.gen_prefix, "scores-")
    print(f"Output file: {args.output_file}")

    data, response = json.load(open(args.input_file)), json.load(open(args.resp_file))
    mode = "descriptive" if "descriptive" in args.resp_file.split("-")[-1] else "reasoning"

    if mode == "descriptive":
        from descriptive_utils import (
            build_descriptive_grading_queries,
            get_descriptive_result,
            postprocess_descriptive_grading_queries,
            preprocess_descriptive_grading_queries,
        )

        groups = preprocess_descriptive_grading_queries(data, response)
        queries = build_descriptive_grading_queries(groups)
        combined_queries = []
        prompts = []
        lengths = []
        for i, query in tqdm(enumerate(queries)):
            prompts.append(query["grading_query"])
            lengths.append(len(query["resp_keys"]))
        results = get_descriptive_result(prompts, lengths)
        for i, query in enumerate(queries):
            combined_query = {**query, **results[i]}
            combined_queries.append(combined_query)
        queries = combined_queries
        queries = postprocess_descriptive_grading_queries(queries)
        with open("queries1.json", "w") as f:
            json.dump(queries, f, indent=4)

    elif mode == "reasoning":
        from reasoning_utils import (
            build_reasoning_grading_queries,
            get_reasoning_result,
        )

        queries = build_reasoning_grading_queries(data, response)
        prompt = []
        for figure_id, query in queries.items():
            prompt.append(query["grading_query"])
        exts, scrs = get_reasoning_result(prompt)
        for i, (figure_id, query) in enumerate(queries.items()):
            queries[figure_id]["extracted_answer"] = exts[i]
            queries[figure_id]["score"] = scrs[i]
            queries[figure_id].pop("grading_query")
    with open(args.output_file, "w") as f:
        json.dump(queries, f, indent=4)

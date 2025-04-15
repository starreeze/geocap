import json
import os

from constants import (
    DESCRIPTIVE_GRADING_ICL,
    DESCRIPTIVE_GRADING_PREFIX,
    DESCRIPTIVE_GRADING_QMAP,
    DESCRIPTIVE_RESP_INST,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_rubric(qid):
    instruction = None
    if qid in [1]:
        instruction = DESCRIPTIVE_GRADING_ICL["title"]
    if qid in [2, 3, 4, 5, 6, 7]:
        instruction = DESCRIPTIVE_GRADING_ICL["ocr"]
    if qid in [8, 9, 10, 12, 14, 15, 17, 19]:
        instruction = DESCRIPTIVE_GRADING_ICL["quant"]
    if qid in [11]:
        instruction = DESCRIPTIVE_GRADING_ICL["bool"]
    if qid in [13]:
        instruction = DESCRIPTIVE_GRADING_ICL["enum"]
    if qid in [16]:
        instruction = DESCRIPTIVE_GRADING_ICL["trend"]
    if qid in [18]:
        instruction = DESCRIPTIVE_GRADING_ICL["layout"]
    assert instruction is not None, f"Instruction for qid {qid} is not found."
    return instruction


def get_descriptive_result(prompt, lengths):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen2.5-14B-Instruct", torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-14B-Instruct")
    contents = []
    for x, length in tqdm(zip(prompt, lengths), total=len(prompt)):
        before_example_end, after_example_end = x.split("### Example End ###")
        before_example_end += "\n### Example End ###"
        messages = [
            {"role": "system", "content": before_example_end},
            {"role": "user", "content": after_example_end},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        try:
            content = json.loads(response)
            verify_grading_output(content, length)
        except Exception as e:
            print(e)
            print(f"error response {response}")
            content = build_dummy_output(length)
        contents.append(content)
    return contents


def get_descriptive_result_gpt(client, prompt, length, max_retries=10):
    curr_retries = 0
    max_tokens = 256
    while curr_retries < max_retries:
        try:
            response = (
                client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="gpt-4o-2024-05-13",
                    response_format={"type": "json_object"},
                    n=1,
                    max_tokens=max_tokens,
                    temperature=0,
                    top_p=1,
                    seed=42,
                )
                .choices[0]
                .message.content
            )
            content = json.loads(response)
            verify_grading_output(content, length)
            break
        except Exception as e:
            print(f"Error: {e}")
            if "Unterminated string starting at" in str(e):
                if max_tokens >= 1024:
                    print(f"Failed to get response for prompt: {prompt}")
                    content = build_dummy_output(length)
                    break
                else:
                    max_tokens = min(1024, max_tokens * 2)
                    print(f"Retrying with max_tokens: {max_tokens}")
            curr_retries += 1
    if curr_retries == max_retries:
        print(f"Failed to get response for prompt: {prompt}")
        content = build_dummy_output(length)
    return content


def build_json_keys(length):
    keys = []
    for i in range(1, length + 1):
        keys.append(f"extract_answer_T{i}")
        keys.append(f"score_T{i}")
    return str(keys)


def populate_grading_inputs(batch):
    query = ""
    for i, (_, response, answer) in enumerate(batch):
        curr_query = "T{}:\nResponse {}: {}\nGround Truth {}: {}\n\n".format(
            i + 1, i + 1, response, i + 1, answer
        )
        query += curr_query
    return query


def verify_grading_output(data, length_data):
    for i in range(1, length_data + 1):
        assert f"extract_answer_T{i}" in data, f"extract_answer_T{i} is not found in {data}"
        assert f"score_T{i}" in data, f"score_T{i} is not found in {data}"
        assert data[f"score_T{i}"] in [0, 1], f"score_T{i} is not in [0, 1]"
    return True


def build_dummy_output(length_data):
    data = {}
    for i in range(1, length_data + 1):
        data[f"extract_answer_T{i}"] = "Failed to parse response"
        data[f"score_T{i}"] = -1
    return data


def preprocess_descriptive_grading_queries(input, resp, num_templates=19):
    groups = {i: [] for i in range(1, num_templates + 1)}
    for _, data in input.items():
        figure_id = data["figure_id"]
        qids = data["qids"]
        for i, qid in enumerate(qids):
            resp_key = f"{figure_id}_{i}"
            response = resp[resp_key]["response"]
            answer = data["answers"][i]
            groups[qid].append((resp_key, response, answer))
    return groups


def build_descriptive_grading_queries(groups, nq_per_query=1):
    queries = []
    for qid, data in groups.items():
        for i in range(0, len(data), nq_per_query):
            batch = data[i : i + nq_per_query]
            question = DESCRIPTIVE_GRADING_QMAP[qid]
            json_keys = build_json_keys(len(batch))
            prefix = (
                DESCRIPTIVE_GRADING_PREFIX.replace("<|NUM_TRIPLETS|>", str(len(batch)))
                .replace("<|OVERARCHING_QUESTION|>", question)
                .replace("<|JSON_KEYS|>", json_keys)
            )
            rubric_icl = get_rubric(qid)
            grading_query = prefix + rubric_icl + populate_grading_inputs(batch)
            curr_query = {"resp_keys": [d[0] for d in batch], "grading_query": grading_query}
            queries.append(curr_query)
    return queries


def postprocess_descriptive_grading_queries(queries):
    scores = {}
    for query in queries:
        resp_keys = query["resp_keys"]
        for i, resp_key in enumerate(resp_keys):
            try:
                extracted_answer = query[f"extract_answer_T{i+1}"]
                score = query[f"score_T{i+1}"]
            except Exception as e:
                print(f"Error: {e}")
                extracted_answer = "falied"
                score = -1
            scores[resp_key] = {"resp_id": resp_key, "extracted_answer": extracted_answer, "score": score}
    return scores


def descriptive_query_helper(qid, subplot_loc):
    if qid in [18, 19]:
        return DESCRIPTIVE_RESP_INST[qid]
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            prefix = "For the current plot, "
        else:
            prefix = f"For the subplot at row {subplot_loc[0]} and column {subplot_loc[1]}, "
    elif isinstance(subplot_loc, str):
        prefix = f"For {subplot_loc}, "
    else:
        raise ValueError(f"Invalid subplot_loc: {subplot_loc}")
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)


def build_descriptive_quries(data, image_dir):
    queries = {}
    for _, d in data.items():
        figure_path = os.path.join(image_dir, f"{d['figure_id']}.jpg")
        for i in range(len(d["qids"])):
            question = descriptive_query_helper(d["qids"][i], d["subplot_loc"])
            curr_query = {
                "figure_id": d["figure_id"],
                "figure_path": figure_path,
                "subq_idx": i,
                "qid": d["qids"][i],
                "question": question,
            }
            queries[f"{d['figure_id']}_{i}"] = curr_query
    return queries

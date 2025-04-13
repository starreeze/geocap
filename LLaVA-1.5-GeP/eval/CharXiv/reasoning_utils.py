import json
import os
from copy import deepcopy

from constants import (
    REASONING_GRADING_INST,
    REASONING_GRADING_PREFIX,
    REASONING_RESP_INST,
)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_reasoning_result(prompt):
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen2.5-14B-Instruct", torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-14B-Instruct")
    exts = []
    scrs = []
    with open("CharXiv/results/error_log1.txt", "a") as log_file:
        for x in tqdm(prompt):
            messages = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                },
                {"role": "user", "content": x},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response = response.strip("```json\n").strip("```")
            response = response.replace("[", "{").replace("]", "}")
            if "score" not in response:
                response = response.replace("{\n", "{").replace("\n}", "}").replace(",\n", ",")
                response = (
                    response[: response.find("{") + 1]
                    + '\n"extract_answer":'
                    + response[response.find("{") + 1 :]
                )
                response = response[: response.rfind("}")] + "\n}" + response[response.rfind("}") + 1 :]
                response = (
                    response[: response.rfind(",")] + ', \n"score":' + response[response.rfind(",") + 1 :]
                )

            try:
                response_json = json.loads(response)
                if "extract_answer" in response_json:
                    ext = response_json["extract_answer"]
                elif "extracted_answer" in response_json:
                    ext = response_json["extracted_answer"]
                scr = response_json["score"]
            except Exception as e:

                log_file.write(f"Error parsing response: {e}\n")
                log_file.write(f"Response: {response}\n")
                print(f"Error parsing response: {e}\n")
                print(f"Response: {response}\n")

                ext, scr = "Failed to parse response", -1
            exts.append(ext)
            scrs.append(scr)
    return exts, scrs


def get_reasoning_result_gpt(client, prompt, max_retries=10):
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
            ext, scr = content["extracted_answer"], content["score"]
            break
        except Exception as e:
            print(f"Error: {e}")
            if "Unterminated string starting at" in str(e):
                if max_tokens >= 1024:
                    print(f"Failed to get response for prompt: {prompt}")
                    ext, scr = "Failed to parse response", -1
                    break
                else:
                    max_tokens = min(1024, max_tokens * 2)
                    print(f"Retrying with max_tokens: {max_tokens}")
            curr_retries += 1
    if curr_retries == max_retries:
        print(f"Failed to get response for prompt: {prompt}")
        ext, scr = "Failed to parse response", -1
    return ext, scr


def get_number_instruction(answer):
    base = answer.split(".")
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    if whole is not None and decimal is None:
        inst = "* Your final answer must be an exact integer."
    elif whole is not None and decimal is not None:
        num_decimal = len(decimal)
        inst = f"* Your final answer must be a number with {num_decimal} decimal places."
    else:
        raise ValueError(f"Invalid answer: {answer}")
    return inst


def build_reasoning_grading_queries(input, resp):
    queries = {}
    for _, data in input.items():
        figure_id = str(data["figure_id"])
        query, response = resp[figure_id]["raw_question"], resp[figure_id]["response"]
        grading_query = REASONING_GRADING_PREFIX + deepcopy(
            REASONING_GRADING_INST[data["inst_category"]]
        ).replace("<|question|>", query).replace("<|ground_truth|>", data["answer"]).replace(
            "<|response|>", response
        )
        query = {"figure_id": figure_id, "grading_query": grading_query}
        queries[figure_id] = query
    return queries


def build_reasoning_queries(data, image_dir):
    queries = {}
    for _, d in data.items():
        figure_path = os.path.join(image_dir, f"{d['figure_id']}.jpg")
        inst_category = d["inst_category"]

        if inst_category in [1, 2, 3]:
            question = REASONING_RESP_INST[inst_category].format(d["query"])

        elif inst_category == 4:
            question = REASONING_RESP_INST[inst_category].format(
                d["query"], get_number_instruction(d["answer"])
            )
        else:
            raise ValueError(f"Invalid instruction category: {inst_category}")
        query = {
            "figure_id": d["figure_id"],
            "figure_path": figure_path,
            "inst_category": inst_category,
            "raw_question": d["query"],
            "question": question,
        }
        queries[d["figure_id"]] = query
    return queries

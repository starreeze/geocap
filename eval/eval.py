import json
import os

from tqdm import tqdm

from common.args import fossil_eval_args, logger
from common.llm import generator_mapping, model_path_mapping
from eval.cal_score import statistics
from eval.utils import (
    calculate_axis_shape_rating,
    calculate_chomata_rating,
    calculate_score,
    characteristics,
    extract_range_or_num,
    extract_tunnel_shape,
    find_first_json_block,
    rule_based_eval_features,
)


def record_err(input, output, error, idx, mode):
    import time

    with open(f"{fossil_eval_args.eval_result_dir}/Error_log.log", "a") as log:
        log.write(
            f"\n{time.ctime()} - {error} @ entry[{idx}] with mode {mode}, examine:\nInput:{input}\nOutput:{output}\n"
        )


class Evaluater:
    def __init__(self) -> None:
        self.loaded_llm = False

    def load_llm_generator(self):
        """Initialize the LLM generator"""
        assert not self.loaded_llm
        # Initialize llm
        model_name, model_id = fossil_eval_args.eval_llm.split("-", 1)  # qwen25-14b
        model_path = model_path_mapping[model_name].format(model_id)
        if model_name != "api":
            self.mode = "local"
        else:
            self.mode = "api"
        self.sys_prompt = "You are a helpful assistant and always respond in JSON format."
        self.llm_generator = generator_mapping[model_name](
            model_path, temperature=0.2, max_tokens=8192, sys_prompt=self.sys_prompt
        )
        self.loaded_llm = True

        # Initialize prompt
        with open("eval/prompts/extract_system_prompt.txt", "r", encoding="utf8") as system_prompt_file:
            self.prompt = system_prompt_file.read()

    def reload_eval_mode(self):
        with open("eval/prompts/eval_system_prompt.txt", "r", encoding="utf8") as system_prompt_file:
            self.prompt = system_prompt_file.read()

    def get_messages(self, testee_batch):
        if self.mode == "local":
            messages = [
                [
                    {"role": "system", "content": self.sys_prompt},
                    {"role": "user", "content": self.prompt.replace("{input}", testee)},
                ]
                for testee in testee_batch
            ]
        else:
            messages = [
                [{"role": "user", "content": self.prompt.replace("{input}", testee)}]
                for testee in testee_batch
            ]
        return messages

    def extract(self, entry_batch, mode):
        fail_flag = False
        testee = [entry[mode] for entry in entry_batch]
        messages = self.get_messages(testee)
        responses = self.llm_generator(messages)
        outputs = []
        for idx, batch in tqdm(enumerate(responses), total=len(messages), desc=f"Eval: Extracting {mode}"):
            try:
                process_json_batch = []
                for response in batch:
                    json_block, remaining_text = find_first_json_block(response)
                    process_json_batch.append(json.loads(json_block))

                # Add image_path key and keep the original order
                for json_item in process_json_batch:
                    new_json_item = {"image_path": entry_batch[idx].get("img", "")}
                    for key, value in json_item.items():
                        new_json_item[key] = value
                    json_item.clear()
                    json_item.update(new_json_item)

            except Exception as e:
                process_json_batch = [{"image_path": entry_batch[idx].get("img", "")}]
                record_err(messages[idx], batch[0], e, idx, mode)
                fail_flag = True
            finally:
                outputs.extend(process_json_batch)

        return outputs, fail_flag

    def make_eval_prompt(self, eval_pair):
        one_eval_pair = []
        for char in characteristics:
            if char in rule_based_eval_features:
                continue

            try:
                A_content = eval_pair[0][char]
            except KeyError:
                A_content = ""
            try:
                B_content = eval_pair[1][char]
            except KeyError:
                B_content = ""
            one_eval_pair.append(f"- {char}\nGenerated:{A_content}\nReference:{B_content}")
        return "\n".join(one_eval_pair)

    def evaluate(self, outputs: list[dict], references: list[dict], caption_batch: list[dict]):
        assert len(outputs) == len(references)
        fail_flag = False
        detailed_scores = []

        prompts = [self.make_eval_prompt(eval_pair) for eval_pair in zip(outputs, references)]
        prompts = self.get_messages(prompts)
        responses = self.llm_generator(prompts, batch_size=1)

        for idx, batch in tqdm(enumerate(responses), total=len(prompts), desc="Eval: Evaluating"):
            score_dict = {"image_path": caption_batch[idx].get("img", "")}
            for char in characteristics:
                score_dict[char] = {"reason": "", "rating": 0}

            try:
                for response in batch:
                    json_block, remaining_text = find_first_json_block(response)
                    detailed_score = json.loads(json_block)
                    for char in detailed_score.keys():
                        score_dict[char] = detailed_score[char]

                for char in characteristics:
                    if len(references[idx].get(char, "")) == 0:
                        score_dict[char] = {"reason": "Reference is empty, skipped evaluation", "rating": -1}
            except Exception as e:
                record_err(prompts[idx], batch[0], e, idx, "evaluation")
                fail_flag = True
            detailed_scores.append(score_dict)

        return detailed_scores, fail_flag


def main():
    evaluater = Evaluater()
    evaluater.load_llm_generator()
    with open(fossil_eval_args.eval_origin_file, "r", encoding="utf-8") as f:  # load to verify the data
        caption_batch = json.load(f)
        caption_batch = caption_batch[fossil_eval_args.eval_start_pos : fossil_eval_args.eval_end_pos]
    if not os.path.exists(fossil_eval_args.eval_result_dir):
        os.makedirs(fossil_eval_args.eval_result_dir, exist_ok=True)

    # Read reference features info
    if not os.path.exists(fossil_eval_args.eval_reference_file):
        ex_r, fail = evaluater.extract(caption_batch, mode="reference")
        with open(fossil_eval_args.eval_reference_file, "w") as f:
            json.dump(ex_r, f, indent=4)
        if fail:
            print(
                "Fail Detected, check log file; program aborted due to unabling to carry on until this error is fixed manually"
            )
            return
    else:
        with open(fossil_eval_args.eval_reference_file, "r") as f:
            ex_r = json.load(f)
        logger.info(f"Loaded reference features info from {fossil_eval_args.eval_reference_file}")

    # Read output features info
    if fossil_eval_args.read_extractions_from_file:
        with open(f"{fossil_eval_args.eval_result_dir}/extracted_output_info.json", "r") as f:
            ex_o = json.load(f)
    else:
        # Extract output feature info
        ex_o, fail = evaluater.extract(caption_batch, mode="output")
        with open(f"{fossil_eval_args.eval_result_dir}/extracted_output_info.json", "w") as f:
            json.dump(ex_o, f, indent=4)
        if fail:
            print("Fail Detected, check log file; carry on to independent reference extraction")
            return

    evaluater.reload_eval_mode()
    assert len(ex_r) == len(
        ex_o
    ), f"Failed extraction valid test, some extractions are not at correct length: ex_o:{len(ex_o)}, ex_r:{len(ex_r)}, caption_batch:{len(caption_batch)}"
    if fossil_eval_args.extract_only:
        return

    # Evaluation
    # detailed, fail = evaluater.evaluate(ex_o, ex_r, caption_batch)
    # with open(f"{fossil_eval_args.eval_result_dir}/detailed_score_list.json", "w") as f:
    #     json.dump(detailed, f, indent=4)

    # ---------debug---------
    with open(f"{fossil_eval_args.eval_result_dir}/detailed_score_list.json", "r") as f:
        detailed = json.load(f)
        fail = False

    # Replace the numerical features with manually caculated ones
    detailed = rule_based_eval(detailed, ex_o, ex_r)

    with open(f"{fossil_eval_args.eval_result_dir}/detailed_score_list.json", "w") as f:
        json.dump(detailed, f, indent=4)
    if fail:
        print("Fail Detected, check log file; program aborted")
        return
    statistics()


def rule_based_eval(detailed_score_list, extracted_output_info, extracted_reference_info):
    new_detailed = []
    for detail, output, reference in zip(
        detailed_score_list, extracted_output_info, extracted_reference_info
    ):
        new_detail = detail.copy()

        for feature in rule_based_eval_features:
            if new_detail[feature].get("rating") == -1:
                continue

            # Calculate scores for numerical features
            if feature in ["length", "width", "ratio", "number_of_volutions", "proloculus", "tunnel_angles"]:
                if not isinstance(reference[feature], str):
                    reference[feature] = str(reference[feature])
                if not isinstance(output[feature], str):
                    output[feature] = str(output[feature])
                ref_range = extract_range_or_num(reference[feature])
                pred_range = extract_range_or_num(output[feature])

                score = calculate_score(ref_range, pred_range)
                new_detail[feature][
                    "reason"
                ] = f"Rule-based eval with output:{output[feature]}->{pred_range}, reference:{reference[feature]}->{ref_range}"
                new_detail[feature]["rating"] = score

            # Calculate scores for chomata
            elif feature == "chomata":
                rating = calculate_chomata_rating(output["chomata"], reference["chomata"])
                new_detail["chomata"]["rating"] = rating
                new_detail["chomata"][
                    "reason"
                ] = f"Rule-based eval with output:{output['chomata']}, reference:{reference['chomata']}"

            # Calculate scores for tunnel shape
            ### 已使用LLM评估，不使用
            # elif feature == "tunnel_shape":
            #     height_output, width_output = extract_tunnel_shape(
            #         output["tunnel_shape"], default_value="none"
            #     )
            #     height_reference, width_reference = extract_tunnel_shape(
            #         reference["tunnel_shape"], default_value="moderate"
            #     )
            #     rating = 0
            #     if height_reference == height_output:
            #         rating += 5
            #     elif height_output == "none":
            #         rating += 0
            #     elif height_reference == "moderate":
            #         rating += 2

            #     if width_reference == width_output:
            #         rating += 5
            #     elif width_output == "none":
            #         rating += 0
            #     elif width_reference == "moderate":
            #         rating += 2

            #     new_detail["tunnel_shape"]["rating"] = rating
            #     new_detail["tunnel_shape"][
            #         "reason"
            #     ] = f"Rule-based eval with output:{output['tunnel_shape']}, reference:{reference['tunnel_shape']}"

            # Calculate scores for axis shape
            elif feature == "axis_shape":
                rating = calculate_axis_shape_rating(output["axis_shape"], reference["axis_shape"])
                new_detail["axis_shape"]["rating"] = rating
                new_detail["axis_shape"][
                    "reason"
                ] = f"Rule-based eval with output:{output['axis_shape']}, reference:{reference['axis_shape']}"

        new_detailed.append(new_detail)
    return new_detailed


if __name__ == "__main__":
    main()

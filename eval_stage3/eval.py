from common.args import eval_stage3_args
from common.llm import generator_mapping, model_path_mapping
import json
from tqdm import tqdm
import os


def record_err(input, output, error, idx, mode):
    import time

    with open(f"{eval_stage3_args.eval_result_dir}/Error_log.log", "a") as log:
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
        model_name, model_id = eval_stage3_args.eval_llm.split("-", 1)  # qwen25-14b
        model_path = model_path_mapping[model_name].format(model_id)
        self.llm_generator = generator_mapping[model_name](model_path)
        self.loaded_llm = True
        self.sys_prompt = "You are a helpful assistant that always responds in JSON format."
        # Initialize prompt
        with open("eval_stage3/extract_system_prompt.txt", "r") as system_prompt_file:
            self.prompt = system_prompt_file.read()

    def reload_eval_mode(self):
        with open("eval_stage3/eval_system_prompt.txt", "r") as system_prompt_file:
            self.prompt = system_prompt_file.read()

    def __make_prompt(self, testee_batch):
        messages = [
            [
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": self.prompt + "\n" + testee},
            ]
            for testee in testee_batch
        ]
        return messages

    def extract(self, entry_batch, mode):
        fail_flag = False
        testee = [entry[mode] for entry in entry_batch]
        messages = self.__make_prompt(testee)
        responses = self.llm_generator(messages)
        outputs = []
        for idx, batch in tqdm(enumerate(responses), total=len(messages), desc=f"Eval: Extracting {mode}"):
            try:
                process_json_batch = [json.loads(batch_ele) for batch_ele in batch]
            except Exception as e:
                record_err(messages[idx], batch[0], e, idx, mode)
                fail_flag = True
                process_json_batch = [{}]
            finally:
                outputs.extend(process_json_batch)
        return outputs, fail_flag

    def evaluate(self, outputs: list[dict], references: list[dict]):
        characteristics = [
            "overall_size",
            "overall_shape",
            "length",
            "width",
            "ratio",
            "axis_shape",
            "number_of_volutions",
            "thickness_of_spirotheca",
            "height_of_volution",
            "proloculus",
            "tunnel_shape",
            "tunnel_angles",
            "chomata",
            "axial_filling",
        ]

        def make_eval_prompt(eval_pair):
            one_eval_pair = []
            for char in characteristics:
                try:
                    A_content = eval_pair[0][char]
                except:
                    if char == "proloculus":
                        try:
                            A_content = eval_pair[0]["initial_chamber"]
                        except Exception as e:
                            print(e)
                            A_content = ""
                    else:
                        A_content = ""

                try:
                    B_content = eval_pair[1][char]
                except:
                    if char == "proloculus":
                        try:
                            B_content = eval_pair[1]["initial_chamber"]
                        except Exception as e:
                            print(e)
                            B_content = ""
                    else:
                        B_content = ""
                one_eval_pair.append(f"-{char}\nA:{A_content}\nB:{B_content}")
            return "\n".join(one_eval_pair)

        assert len(outputs) == len(references)
        fail_flag = False
        prompts = [make_eval_prompt(eval_pair) for eval_pair in zip(outputs, references)]
        prompts = self.__make_prompt(prompts)
        responses = self.llm_generator(prompts, batch_size=1)
        detailed_scores = []
        for idx, batch in tqdm(enumerate(responses), total=len(prompts), desc=f"Eval: Evaluating"):
            try:
                score_list: list[dict] = [json.loads(batch_ele) for batch_ele in batch]
            except Exception as e:
                record_err(prompts[idx], batch[0], e, idx, "evaluation")
                fail_flag = True
            else:
                detailed_scores.extend(score_list)
        return detailed_scores, fail_flag


def main():
    evaluater = Evaluater()
    evaluater.load_llm_generator()
    with open(eval_stage3_args.eval_origin_file, "r") as f:  # load to verify the data
        caption_batch = json.load(f)
    if not os.path.exists(eval_stage3_args.eval_result_dir):
        os.makedirs(eval_stage3_args.eval_result_dir, exist_ok=True)
    if eval_stage3_args.read_extractions_from_file:
        with open(f"{eval_stage3_args.eval_result_dir}/extracted_output_info.json", "r") as f:
            ex_o = json.load(f)
        with open(f"{eval_stage3_args.eval_result_dir}/extracted_reference_info.json", "r") as f:
            ex_r = json.load(f)
    else:
        ex_o, fail = evaluater.extract(caption_batch, mode="output")
        with open(f"{eval_stage3_args.eval_result_dir}/extracted_output_info.json", "w") as f:
            json.dump(ex_o, f)
        if fail:
            print("Fail Detected, check log file; carry on to independent reference extraction")
            return
        ex_r, fail = evaluater.extract(caption_batch, mode="reference")
        with open(f"{eval_stage3_args.eval_result_dir}/extracted_reference_info.json", "w") as f:
            json.dump(ex_r, f)
        if fail:
            print(
                "Fail Detected, check log file; program aborted due to unabling to carry on until this error is fixed manually"
            )
            return
    evaluater.reload_eval_mode()
    assert (
        len(ex_r) == len(ex_o) and len(ex_r) == len(caption_batch) and len(ex_o) == len(caption_batch)
    ), f"Failed extraction valid test, some extractions are not at correct length: ex_o:{len(ex_o)}, ex_r:{len(ex_r)}"
    detailed, fail = evaluater.evaluate(ex_o, ex_r)
    with open(f"{eval_stage3_args.eval_result_dir}/detailed_score_list.txt", "w") as f:
        json.dump(detailed, f)
    if fail:
        print("Fail Detected, check log file; program aborted")
        return


if __name__ == "__main__":
    main()

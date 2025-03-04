import base64
import json

from tqdm import tqdm

from common.args import fossil_eval_args

# from common.llm import generator_mapping, model_path_mapping
# from common.llm import APIGenerator  # sry but now we do this
from common.vllm.Qwen2_VL_Instruct import GenerateModel  # modify `api` here to change the model


class Generator:
    def __init__(self) -> None:
        self.loaded_llm = False

    def load_llm_generator(self):
        """Initialize the LLM generator"""
        assert not self.loaded_llm
        # Initialize llm
        self.llm_generator = GenerateModel()
        self.loaded_llm = True

    def generate(self, pictures, questions, references) -> list[dict]:
        data = []
        for p, q, r in zip(pictures, questions, references):
            entry = {"img": p, "question": q, "output": "", "reference": r}  # later fill this
            data.append(entry)
        for idx, output in tqdm(
            enumerate(self.llm_generator.generate(pictures, questions)),
            total=len(pictures),
            desc="Generating Descriptions:",
        ):
            data[idx]["output"] = output[0]
        return data


def main():
    sys_prompt = "You will be provided with a fossil picture input and you should depict this picture. Do not use sub-titles and lists to devide the output; instead, merge the paragraph into a whole. You do not need a conclusion, and you should try to mimic the example. Here is an example:\nUser:The following is an image of a paleontological fossil, please provide a detailed description for the fossil image. Here is some information about the fossil:\nlength: 10.802 mm. , width(diameter): 3.179 mm. ratio: 3.398\nnumber of volutions(whorls): 6.5\naverage thickness of spirotheca: 35 microns\nthickness by volutions: 21, 34, 40, 30, 52, 30, 40 microns\nheights of volution/whorl: 157, 170, 185, 272, 285, 292 microns\ninitial chamber(proloculus): 219 microns\ntunnel angles: 20 degrees in the 2nd volution, 24 degrees in the 5th volution.\nAssistant:Test large, rhomboid in outline, concave in middle on one side and convex on another side; median axis sinuous, and extremities sharp. Volutions 8-10; inner 4-5 volutions tightly involute and successive ones loosing gradually. Holotype specimen 10.87 mm long, 2.92 mm wide, and about 3.72:1 in axial ratio. Spirotheca very thin in tightly coiled inner volutions, about 0.008-0.01 mm thick; thickening outwards; composed of a tectum and a keriotheca; the spirotheca on the eighth volution about 0.07 mm thick. Septa are plane and straight in the inner volutions, and completely fluted in the outer volutions; flutings broadly rounded in general, sometimes reaching the top of the chambers in height. Chomata small, only visible in the inner volutions. Tunnels low, moderately wide. Axial fillings heavy, distributed on the lateral sides of median axis, except the final volution. Proloculus circular."  # Test: No Pre-info & With Pre-info
    generator = Generator()
    generator.load_llm_generator()
    pics, questions, references = [], [], []
    with open("instructions.jsonl", "r") as f:
        for line in f:
            data = json.loads(line)
            pics.append(
                f"/home/nfs05/xiangch/geocap/dataset/common/filtered_images/{data['image']}"
            )  # No not here...
            questions.append(sys_prompt + data["input"])
            references.append(data["output"])
    data = generator.generate(pics, questions, references)
    with open(fossil_eval_args.eval_origin_file, "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()

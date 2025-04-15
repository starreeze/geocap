from typing import Any
import os
import base64
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import argparse
import time
import json
import jsonlines


class MyQwenVLGenerator:
    def __init__(self, model: str, **kwargs) -> None:
        self.min_pixels = 256 * 28 * 28
        self.max_pixels = 256 * 28 * 28
        self.processor = AutoProcessor.from_pretrained(
            model, min_pixels=self.min_pixels, max_pixels=self.max_pixels
        )
        self.temperature = kwargs.get("temperature", 0.2)
        self.max_tokens = kwargs.get("max_tokens", 512)
        self.sys_prompt = kwargs.get("sys_prompt", "You are a helpful assistant.")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model, device_map="auto", torch_dtype="auto"
        )

    def construct_input(self, mixed_inputs: list[tuple[str, str]]):
        content = []
        for input_type, data in mixed_inputs:
            if input_type == "text":
                content.append({"type": "text", "text": data})
            elif input_type == "image":
                if os.path.exists(data):
                    with open(data, "rb") as f:
                        data = base64.b64encode(f.read()).decode("utf-8")
                content.append(
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{data}",
                        "min_pixels": self.min_pixels,
                        "max_pixels": self.max_pixels,
                    }
                )
        messages = [{"role": "system", "content": self.sys_prompt}, {"role": "user", "content": content}]

        return messages

    def generator(self, batch, batch_size):
        texts = [
            self.processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, add_vision_id=True
            )
            for msg in batch
        ]
        image_inputs, video_inputs = process_vision_info(batch)
        inputs = self.processor(
            text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        inputs = inputs.to("cuda")

        # Batch Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts

    def __call__(self, inputs: list[list[tuple[str, str]]], batch_size=1):
        input_texts = inputs
        target_range = range((len(input_texts) + batch_size - 1) // batch_size)
        for i in target_range:
            results: list[str] = []
            batch = input_texts[i * batch_size : (i + 1) * batch_size]
            batch = [self.construct_input(inputx) for inputx in batch]
            outputs = self.generator(batch, batch_size=batch_size)
            for output in outputs:  # type: ignore
                generated_text = output.strip()
                results.append(generated_text)
            yield results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str, required=True)
    parser.add_argument("--start_pos", type=int, default=0)
    parser.add_argument("--end_pos", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.eval_path + "_eval"):
        os.makedirs(args.eval_path + "_eval")
    pics = os.listdir(args.eval_path)
    pics.sort(key=lambda x: int(x.split(".")[0]))
    if args.end_pos != -1:
        pics = pics[args.start_pos : args.end_pos]
    else:
        pics = pics[args.start_pos :]
    user_sys_prompt_front = """## Task Description
You will receive input as an fossil image, You need to score that image in "Your turn" section, where:

- **1** = The image quality is very poor, for reasons such as:

* The shell ring is incomplete, with large white gaps.
* The shell ring is not a complete geometric shape (such as an ellipse, fusiform or spindle shape).
* There is an abnormal indentation on the left or right side of the outermost shell ring.
* Another complete fossil image is included on the left or right side inside the fossil image. 
* The axial accumulation (the large black area near the horizontal symmetry axis of the image) exists but does not extend to both sides along the horizontal symmetry axis; instead, it abruptly appears on both sides.
* The folds (the irregular porous structures on the horizontal sides of the image) do not fully fill the outermost shell ring in the vertical direction.
* The image contains only clean concentric geometric shell rings, with no other feature points, except at the horizontal ends.

- **2** = The image quality is average, for reasons such as:

* None of the issues of the 1-point image are clearly present.
* The shell ring is not clear and appears blurry.
* The folds (the irregular porous structures on the horizontal sides of the image) are not distinct.

- **3** = The image quality is good, for reasons such as:

* None of the issues of the 1-point and 2-point images are present.

- **4** = The image is a real fossil image, used to help you understand the above scoring criteria.

## Instructions
1. Analyze the given fossil images.
2. Assign a score (0-3, only integer allowed) based on how well the quality of the unscored image is based on the criteria above.
3. Output a valid JSON object with "reason" and "score" as keys and your reasoning and assigned score as values, respectively.

## Response requirements
First elaborate your analysis and reasoning, then provide the final score, an integer between 1 (worst) and 3 (best). You should put your analysis and reasoning in "reason" part whilst put your final score into "score" part, as the following format suggests:

{"reason": "your reasoning here", "score": 2}

You should not provide user with extra content such as 'Here's the analysis and score for the image:', etc.

## Example:

    """
    pic_prefix = "/home/nfs04/xingsy/sam/geocap/dataset/"
    real_prefix = "/home/nfs04/xingsy/foscap/dataset/common/images/"
    examples = [
        # (real_prefix+"Chusenella_ellipsoidalis_1_1.png",10),
        (
            real_prefix + "Fusulina_ellipsoformis_1_1.png",
            json.dumps(
                {
                    "reason": "This fossil image features clear lines, with all whorls being spindle-shaped and remarkably intact. The whorls exhibit a rich variety of solid and hollow geometric patterns. The hollow patterns on the left and right sides of the image fill the blank spaces of the outermost whorl vertically, respectively. The entire image shows no abnormal protrusions or missing parts in any of the whorls.",
                    "score": 4,
                }
            ),
        ),
        # (real_prefix+"Fusulinella_thompsoni_1_2.png",10),
        # (pic_prefix+"stage2_100k_pics_part1/00000001.jpg",3),
        # (pic_prefix+"stage2_100k_pics_part1/00000004.jpg",1),
        (
            pic_prefix + "stage2_100k_pics_part1/00000003.jpg",
            json.dumps(
                {
                    "reason": "The lower left corner of this image contains another complete fossil specimen within it.",
                    "score": 1,
                }
            ),
        ),
        (
            pic_prefix + "stage2_100k_pics_part1/00000038.jpg",
            json.dumps(
                {
                    "reason": "The fossil in this image has relatively clear outlines, with abundant solid and hollow geometric patterns on its whorls. Most of the whorls show no abnormal protrusions or missing sections, but the lower right corner of the outermost whorl is fractured.",
                    "score": 2,
                }
            ),
        ),
        # (pic_prefix+"stage2_100k_pics_part1/00000038.jpg",6),
        # (pic_prefix+"stage2_100k_pics_part1/00000073.jpg",6),
        (
            pic_prefix + "stage2_100k_pics_part1/00000129.jpg",
            json.dumps(
                {
                    "reason": "This image exhibits clear outlines with intact whorls displaying rich characteristic patterns. The black axial deposits extend symmetrically from the central horizontal axis towards both sides. The whorls show no abnormal fractures or protrusions, and no other fossil specimens are contained within them.",
                    "score": 3,
                }
            ),
        ),
    ]
    user_prompt = [("text", user_sys_prompt_front)]
    for i in range(len(examples)):
        user_prompt.append(("text", f"Example{i+1}:"))
        user_prompt.append(("image", examples[i][0]))
        # user_prompt.append(("text","User: Please rate this fossil image."))
        user_prompt.append(("text", f"Output: {examples[i][1]}"))
    final_inputs = []
    for pic in pics:
        final_inputs.append(
            user_prompt
            + [
                ("text", "Your turn:"),
                ("image", f"{args.eval_path}/{pic}"),
                # ("text","User: Please rate this fossil image."),
                ("text", f"Output: "),
            ]
        )
    # generator = MyQwenVLGenerator(model="/home/nfs02/model/Qwen2.5-VL-7B-Instruct")
    generator = MyQwenVLGenerator(model="/home/nfs05/model/Qwen_Qwen2.5-VL-32B-Instruct")
    result_generator = generator(final_inputs, batch_size=args.batch_size)
    outputs = []
    total_batches = (len(final_inputs) + args.batch_size - 1) // args.batch_size
    os.makedirs(f"{os.path.join(args.save_path,args.eval_path.split('/')[-1])}_eval", exist_ok=True)
    index = 0
    with jsonlines.open(
        f"{os.path.join(args.save_path,args.eval_path.split('/')[-1])}_eval/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json",
        "a",
        flush=True,
    ) as f:
        for batch in tqdm(result_generator, total=total_batches, desc="Evaluating"):
            outputs.extend(batch)
            for i in range(len(batch)):
                f.write({pics[index + i]: batch[i]})
            index += len(batch)

    # with open(f"{os.path.join(args.save_path,args.eval_path.split('/')[-1])}_eval/{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.json","w") as f:
    #     result_dict = {}
    #     for i in range(len(pics)):
    #         result_dict[pics[i]]=outputs[i]
    #     json.dump(result_dict,f,ensure_ascii=False,indent=4)


if __name__ == "__main__":
    main()

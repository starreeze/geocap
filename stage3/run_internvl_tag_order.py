import json
import math
import os

import torch
import torchvision.transforms as T
from internvl_chat.internvl.conversation import get_conv_template
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        "InternVL2_5-1B": 24,
        "InternVL2_5-2B": 24,
        "InternVL2_5-4B": 32,
        "InternVL2_5-8B": 32,
        "InternVL2_5-26B": 48,
        "InternVL2_5-38B": 60,
        "InternVL2_5-78B": 80,
    }[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


def main():
    path = "internvl_chat/work_dirs/foscap/stage3_only_latest"
    output_path = "./outputs/stage3_only_latest.jsonl"

    test_path = "/home/nfs05/xiangch/geocap/dataset/latest/stage3_internvl_test.jsonl"
    image_root = "/home/nfs05/xiangch/geocap/dataset/common/images/"

    # device_map = split_model('InternVL2_5-8B')
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="cuda",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=2048, do_sample=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    test_data = []
    with open(test_path, "r") as f_in:
        for line in f_in:
            data = json.loads(line)
            test_data.append(data)

    all_tags = [
        "<shell>",
        "<length>",
        "<width>",
        "<ratio>",
        "<volution>",
        "<proloculus>",
        "<axis>",
        "<axial filling>",
        "<spirotheca>",
        "<septa>",
        "<chomata>",
        "<tunnel shape>",
        "<tunnel angle>",
    ]

    with open(output_path, "w") as f_out:
        for data in tqdm(test_data):
            image_file = os.path.join(image_root, data["image"])
            pixel_values = load_image(image_file, max_num=6).to(torch.bfloat16).cuda()
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []

            question = data["conversations"][0]["value"]
            response = ""

            # Generate until all tags are in the response in order
            # Track the next tag we need to ensure appears in the response
            for tag in all_tags:
                eos_token_id = tokenizer.convert_tokens_to_ids(["<", "</", ".</", "</s"])
                generation_config["eos_token_id"] = eos_token_id

                template = get_conv_template(model.template)
                template.system_message = model.system_message
                template.append_message(template.roles[0], question)
                template.append_message(template.roles[1], response + f" {tag}")
                query = template.get_prompt()
                # Remove the trailing <|im_end|>\n
                assert query.endswith("<|im_end|>\n")
                query = query[: -len("<|im_end|>\n")]

                for num_patches in num_patches_list:
                    image_tokens = "<img>" + "<IMG_CONTEXT>" * model.num_image_token * num_patches + "</img>"
                    query = query.replace("<image>", image_tokens, 1)

                model_inputs = tokenizer(query, return_tensors="pt")
                input_ids = model_inputs["input_ids"].to("cuda")

                cur_response_ids = model.generate(pixel_values, input_ids, **generation_config)
                cur_response = tokenizer.batch_decode(cur_response_ids, skip_special_tokens=True)[0]
                cur_response = cur_response.split("<")[0]
                response = response + tag + cur_response + tag.replace("<", "</")

            output_data = {"image": data["image"], "question": question, "response": response}
            f_out.write(json.dumps(output_data, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

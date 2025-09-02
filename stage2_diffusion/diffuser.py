import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.append(".")
sys.path.append("..")

import torch
import torch.nn.functional as F
from safetensors.numpy import save_file, load_file
from omegaconf import OmegaConf
from transformers import AutoConfig
import cv2
from PIL import Image
import numpy as np
import json
import os

#
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipelineLegacy,
    StableDiffusionInpaintPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor

#
from models.pipeline_mimicbrush import MimicBrushPipeline
from models.ReferenceNet import ReferenceNet
from models.depth_guider import DepthGuider
from mimicbrush import MimicBrush_RefNet
from dataset.data_utils import *

val_configs = OmegaConf.load("./configs/inference.yaml")

# === import Depth Anything ===
import sys
sys.path.append("../depthanything")
sys.path.append("/home/nfs03/xingsy/MimicBrush/depthanything")

from torchvision.transforms import Compose
from depthanything.fast_import import depth_anything_model
from depthanything.depth_anything.util.transform import (
    Resize,
    NormalizeImage,
    PrepareForNet,
)

transform = Compose(
    [
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ]
)
depth_anything_model.load_state_dict(torch.load(val_configs.model_path.depth_model))


# === load the checkpoint ===
print("\033[1;33mLoading checkpoint...\033[0m")

base_model_path = val_configs.model_path.pretrained_imitativer_path
vae_model_path = val_configs.model_path.pretrained_vae_name_or_path
image_encoder_path = val_configs.model_path.image_encoder_path
ref_model_path = val_configs.model_path.pretrained_reference_path
mimicbrush_ckpt = val_configs.model_path.mimicbrush_ckpt_path
device = "cuda"

print("\033[1;32mCheckpoint loaded.\033[0m")

# === load the modules (copy from gradio demo) ===
print("\033[1;33mLoading modules...\033[0m")
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
unet = UNet2DConditionModel.from_pretrained(
    base_model_path,
    subfolder="unet",
    in_channels=13,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
).to(dtype=torch.float16)

pipe = MimicBrushPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    unet=unet,
    feature_extractor=None,
    safety_checker=None,
)

depth_guider = DepthGuider()
referencenet = ReferenceNet.from_pretrained(ref_model_path, subfolder="unet").to(
    dtype=torch.float16
)
mimicbrush_model = MimicBrush_RefNet(
    pipe,
    image_encoder_path,
    mimicbrush_ckpt,
    depth_anything_model,
    depth_guider,
    referencenet,
    device,
)
mask_processor = VaeImageProcessor(
    vae_scale_factor=1, do_normalize=False, do_binarize=True, do_convert_grayscale=True
)
print("\033[1;32mModules loaded, we are ready to go.\033[0m")
# === Main ===

def pad_img_to_square(original_image, is_mask=False):
    width, height = original_image.size

    if height == width:
        return original_image

    if height > width:
        padding = (height - width) // 2
        new_size = (height, height)
    else:
        padding = (width - height) // 2
        new_size = (width, width)

    if is_mask:
        new_image = Image.new("RGB", new_size, "black")
    else:
        new_image = Image.new("RGB", new_size, "white")

    if height > width:
        new_image.paste(original_image, (padding, 0))
    else:
        new_image.paste(original_image, (0, padding))
    return new_image

def collage_region(low, high, mask):
    mask = (np.array(mask) > 128).astype(np.uint8)
    low = np.array(low).astype(np.uint8)
    low = (low * 0).astype(np.uint8)
    high = np.array(high).astype(np.uint8)
    mask_3 = mask
    collage = low * mask_3 + high * (1 - mask_3)
    collage = Image.fromarray(collage)
    return collage


def resize_image_keep_aspect_ratio(image, target_size=512):
    height, width = image.shape[:2]
    if height > width:
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = int(height * (target_size / width))
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def infer_single(
    ref_image,
    target_image,
    target_mask,
    seed=-1,
    num_inference_steps=50,
    guidance_scale=5.0,
    enable_shape_control=False,
):
    # return ref_image
    """
    mask: 0/1 1-channel  np.array
    image: rgb           np.array
    """
    ref_image = ref_image.astype(np.uint8)
    target_image = target_image.astype(np.uint8)
    target_mask = target_mask.astype(np.uint8)

    ref_image = Image.fromarray(ref_image.astype(np.uint8))
    ref_image = pad_img_to_square(ref_image)

    target_image = pad_img_to_square(Image.fromarray(target_image))
    target_image_low = target_image

    target_mask = (
        np.stack([target_mask, target_mask, target_mask], -1).astype(np.uint8) * 255
    )
    target_mask_np = target_mask.copy()
    target_mask = Image.fromarray(target_mask)
    target_mask = pad_img_to_square(target_mask, True)

    target_image_ori = target_image.copy()
    target_image = collage_region(target_image_low, target_image, target_mask)

    depth_image = target_image_ori.copy()
    depth_image = np.array(depth_image)
    depth_image = transform({"image": depth_image})["image"]
    depth_image = torch.from_numpy(depth_image).unsqueeze(0) / 255

    if not enable_shape_control:
        depth_image = depth_image * 0

    mask_pt = mask_processor.preprocess(target_mask, height=512, width=512)

    pred, depth_pred = mimicbrush_model.generate(
        pil_image=ref_image,
        depth_image=depth_image,
        num_samples=1,
        num_inference_steps=num_inference_steps,
        seed=seed,
        image=target_image,
        mask_image=mask_pt,
        strength=1,
        prompt="Monochrome, fossil, ancient",
        guidance_scale=guidance_scale,
    )

    depth_pred = F.interpolate(
        depth_pred, size=(512, 512), mode="bilinear", align_corners=True
    )[0][0]
    depth_pred = (
        (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min()) * 255.0
    )
    depth_pred = depth_pred.detach().cpu().numpy().astype(np.uint8)
    depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)[:, :, ::-1]

    pred = pred[0]
    pred = np.array(pred).astype(np.uint8)
    return pred, depth_pred.astype(np.uint8)

def crop_padding_and_resize(ori_image, square_image):
    ori_height, ori_width, _ = ori_image.shape
    scale = max(ori_height / square_image.shape[0], ori_width / square_image.shape[1])
    resized_square_image = cv2.resize(
        square_image,
        (int(square_image.shape[1] * scale), int(square_image.shape[0] * scale)),
    )
    padding_size = max(
        resized_square_image.shape[0] - ori_height,
        resized_square_image.shape[1] - ori_width,
    )
    if ori_height < ori_width:
        top = padding_size // 2
        bottom = resized_square_image.shape[0] - (padding_size - top)
        cropped_image = resized_square_image[top:bottom, :, :]
    else:
        left = padding_size // 2
        right = resized_square_image.shape[1] - (padding_size - left)
        cropped_image = resized_square_image[:, left:right, :]
    return cropped_image


def processRefImage(ref_image):
    height, width = ref_image.shape[:2]

    # 计算中间位置的x坐标
    center_x = width // 2

    # 定义线的起点和终点坐标
    # 起点: (center_x, 0) - 图像顶部
    # 终点: (center_x, height-1) - 图像底部
    start_point = (center_x, 0)
    end_point = (center_x, height - 1)

    # 定义线的颜色 (BGR格式)，默认为红色
    color = (255, 255, 255)

    # 定义线的粗细，默认为1像素
    thickness = 50

    # 在图像上绘制线
    image_with_line = cv2.line(ref_image, start_point, end_point, color, thickness)

    return image_with_line


def get_random_match(dir: str):
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    lucky_guy = random.choice(files)
    return lucky_guy


def diffuse(
    img: np.ndarray,
    mask: np.ndarray,
    best_ref_poles: dict,
    ref_path: str,
    num_refs: int,
    mode: str,
    debug=None,
) -> np.ndarray:

    def get_random_ref_image(ref_path: list) -> np.ndarray:
        ref_image = cv2.imread(random.choice(ref_path))
        return ref_image

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
    pil_img = pil_img.convert("L").convert("RGB")
    img = np.asarray(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pil_mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGRA2RGB))
    pil_mask = pil_mask.convert("L")

    mask = np.asarray(pil_mask)
    mask = 255 - mask
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)
    mask[:, 0] = 0
    mask[0, :] = 0
    mask[mask.shape[0]-1, :] = 0
    mask[:, mask.shape[0]-1] = 0

    if np.all(mask == 0):
        return img.copy() # No need to diffuse

    if mode == "axial_main":
        ref_paths = [os.path.join(ref_path, f"ref_axial_{x:08}.png") for x in range(num_refs)]
        ref_image = get_random_ref_image(ref_paths)
    elif mode == "axial_ext":
        ref_paths = [os.path.join(ref_path, f"ref_axial_ext_{x:08}.png") for x in range(num_refs)]
        ref_image = get_random_ref_image(ref_paths)
    elif mode == "poles":
        # ref_paths = [os.path.join(ref_path, f"ref_{x:08}.png") for x in range(num_refs)]
        # ref_image = get_best_match(img, ref_paths)
        ref_image = cv2.imread(best_ref_poles["best_ref"], cv2.IMREAD_UNCHANGED)
        transparent_mask = ref_image[:, :, 3] == 0  # alpha通道为0的像素
        ref_image[transparent_mask] = [255, 255, 255, 255]  # 一次性设置RGBA
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGBA2RGB)
        target_width = abs(best_ref_poles["bbox"][0][0] - best_ref_poles["bbox"][1][0])
        target_height = abs(best_ref_poles["bbox"][0][1] - best_ref_poles["bbox"][1][1])
        original_height, original_width = ref_image.shape[:2]
        original_ratio = original_width / original_height
        target_ratio = target_width / target_height

        # 计算新尺寸
        if original_ratio < target_ratio:
            # 原始图像更"瘦高"，需要放大宽度
            new_width = int(original_height * target_ratio)
            new_height = original_height
        else:
            # 原始图像更"矮胖"，需要放大高度
            new_width = original_width
            new_height = int(original_width / target_ratio)

        # 使用INTER_CUBIC插值进行放大（保持高质量）
        ref_image = cv2.resize(ref_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        ref_image = processRefImage(ref_image)
    
    if debug is not None:
        cv2.imwrite(f"{debug}_Image_{mode}.png", img)
        cv2.imwrite(f"{debug}_Reference_{mode}.png", ref_image)

    synthesis, depth_pred = infer_single(
        ref_image.copy(), img.copy(), mask.copy(), num_inference_steps=60, guidance_scale=6, seed=0, enable_shape_control=True
    )

    if debug is not None:
        cv2.imwrite(f"{debug}_SYNTH_{mode}_before_edit.png", synthesis)

    synthesis = crop_padding_and_resize(img, synthesis)
    mask_3 = np.stack([mask, mask, mask], -1).astype(np.uint8) * 255
    mask_alpha = mask_3.copy()
    mask_3_bin = mask_alpha / 255
    synthesis = synthesis * mask_3_bin + img * (1 - mask_3_bin)
    if debug is not None:
        cv2.imwrite(f"{debug}_SYNTH_{mode}.png", synthesis)

    return synthesis.astype(np.uint8)

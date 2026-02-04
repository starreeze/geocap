# -*- coding: utf-8 -*-
# @Date    : 2025-12-15 14:22:28
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations

import io
import os
from typing import Any, Mapping

from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams  # pyright: ignore[reportAttributeAccessIssue]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_path_like(x: Any) -> bool:
    return isinstance(x, str) and len(x.strip()) > 0


def _load_pil_image(image: Any):
    """
    Load an RGB PIL image from:
    - PIL.Image.Image
    - str path
    - bytes (encoded image bytes)
    """
    if image is None:
        raise ValueError("`image` is required for LLaVA inference but got None.")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if _is_path_like(image):
        path = str(image)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image path does not exist: {path}")
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image: {path}") from e

    if isinstance(image, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(bytes(image))).convert("RGB")
        except Exception as e:
            raise RuntimeError("Failed to decode image bytes.") from e

    raise TypeError("Unsupported `image` type. Expected PIL.Image.Image, str path, or bytes.")


class VllmLlava:
    """
    vLLM-backed inference wrapper for **LLaVA-1.5** family models (image + text).

    Each input item must provide:
    - `prompt`: user prompt text (WITHOUT chat template)
    - `image`: image path OR PIL image OR encoded bytes
    Any other fields are preserved in the output unchanged.
    """

    def __init__(
        self,
        hf_model_path: str,
        has_image: bool = True,
        model_image_token: str = "<image>",
        dataset_image_token: str = "<image>",
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = -1,
        tensor_parallel_size: int = 1,
        dtype: str = "half",
        max_model_len: int | None = None,
        gpu_memory_utilization: float = 0.90,
        repetition_penalty: float = 1.0,
        stop: list[str] | None = None,
        trust_remote_code: bool = True,
        enforce_eager: bool = True,
    ):
        _require(
            isinstance(hf_model_path, str) and hf_model_path.strip() != "",
            "`hf_model_path` must be a non-empty HF model id or local path.",
        )
        _require(tensor_parallel_size >= 1, "`tensor_parallel_size` must be >= 1.")
        _require(0.0 < gpu_memory_utilization <= 1.0, "`gpu_memory_utilization` must be in (0, 1].")
        _require(isinstance(max_tokens, int) and max_tokens > 0, "`max_tokens` must be a positive int.")
        _require(isinstance(top_k, int), "`top_k` must be an int.")
        _require(isinstance(temperature, (int, float)) and temperature >= 0.0, "`temperature` must be >= 0.")
        _require(isinstance(top_p, (int, float)) and 0.0 <= float(top_p) <= 1.0, "`top_p` must be in [0, 1].")
        _require(
            isinstance(repetition_penalty, (int, float)) and float(repetition_penalty) > 0.0,
            "`repetition_penalty` must be > 0.",
        )
        if stop is not None:
            _require(
                isinstance(stop, list) and all(isinstance(s, str) for s in stop), "`stop` must be list[str]."
            )
        if has_image:
            _require(
                isinstance(model_image_token, str) and model_image_token.strip() != "",
                "`model_image_token` must be a non-empty string.",
            )
            _require(
                isinstance(dataset_image_token, str) and dataset_image_token.strip() != "",
                "`dataset_image_token` must be a non-empty string.",
            )

        self.hf_model_path = hf_model_path
        self.has_image = has_image
        self.model_image_token = model_image_token
        self.dataset_image_token = dataset_image_token
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.repetition_penalty = float(repetition_penalty)
        self.stop = stop

        # Tokenizer is used only for chat templating (string prompt building).
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=trust_remote_code)

        # vLLM engine.
        llm_kwargs: dict[str, Any] = dict(
            model=hf_model_path,
            tokenizer=hf_model_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )
        if max_model_len is not None:
            _require(max_model_len > 0, "`max_model_len` must be > 0.")
            llm_kwargs["max_model_len"] = int(max_model_len)

        self.llm = LLM(**llm_kwargs)

    def _build_llava_prompt(self, user_prompt_text: str) -> str:
        """
        Build a LLaVA-style chat prompt with an `<image>` token.

        Prefers `tokenizer.apply_chat_template` when available (HF chat template),
        otherwise falls back to a simple 'USER/ASSISTANT' format.
        """
        _require(
            isinstance(user_prompt_text, str) and user_prompt_text.strip() != "",
            "`prompt` must be a non-empty string.",
        )
        _require(
            user_prompt_text.count(self.dataset_image_token) <= 1, "Currently only support single image VQA"
        )

        if self.has_image:
            user_prompt_text = user_prompt_text.replace(self.dataset_image_token, self.model_image_token)
            if self.model_image_token not in user_prompt_text:
                user_prompt_text = f"{self.model_image_token}\n{user_prompt_text}"
            messages = [{"role": "user", "content": [{"type": "text", "text": user_prompt_text}]}]
        else:
            messages = [{"role": "user", "content": user_prompt_text}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if self.has_image:
            assert self.model_image_token in rendered
        return rendered

    def __call__(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Inference with VLLM.

        Args:
            inputs: A batch of inputs to the model. Each item should be a dictionary with:
                - prompt: The complete user prompt text (before applying chat template and w/o <image> token).
                - image: The image path.
                - any other fields (kept unmodified)

        Returns:
            A list of dictionaries (same length/order as `inputs`), each containing:
                - prompt
                - image
                - response: The raw response text from the model
                - any other fields (unmodified)
        """
        # Build vLLM requests.
        requests: list[dict[str, Any]] = []
        for i, item in tqdm(enumerate(inputs), total=len(inputs), desc="Building vLLM requests"):
            if not isinstance(item, Mapping):
                raise TypeError(f"inputs[{i}] must be a dict-like object.")
            if "prompt" not in item:
                raise KeyError(f"inputs[{i}] missing required key: `prompt`.")

            prompt_text = item["prompt"]

            if not isinstance(prompt_text, str) or prompt_text.strip() == "":
                raise TypeError(f"inputs[{i}]['prompt'] must be a non-empty string.")
            prompt_str = self._build_llava_prompt(prompt_text)
            vllm_request: dict = {"prompt": prompt_str}
            if "image" in item:
                _require(self.has_image, "`has_image` must be True when `image` is provided.")
                pil_image = _load_pil_image(item["image"])
                vllm_request["multi_modal_data"] = {"image": pil_image}
            requests.append(vllm_request)

        sp = SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop=self.stop,
        )

        outputs = self.llm.generate(requests, sampling_params=sp)  # type: ignore[arg-type]

        # Map outputs back to inputs (vLLM preserves order).
        results: list[dict[str, Any]] = []
        if len(outputs) != len(inputs):
            raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(inputs)} inputs.")

        for item, out in zip(inputs, outputs, strict=True):
            # vLLM RequestOutput: `.outputs[0].text` is the generated continuation.
            out_outputs = getattr(out, "outputs", None)
            if not out_outputs:
                raise RuntimeError("vLLM returned an output object with no generations (`outputs` is empty).")

            gen_text = getattr(out_outputs[0], "text", None)
            if gen_text is None:
                raise RuntimeError("vLLM returned a generation with `text=None`.")

            merged = dict(item)
            merged["response"] = str(gen_text)
            results.append(merged)

        return results


def main():
    vllm_llava = VllmLlava(
        hf_model_path="/home/nfs05/model/llava-hf/llava-1.5-7b-hf", gpu_memory_utilization=0.7
    )
    inputs = [
        {
            "prompt": "Please describe the image in detail.",
            "image": "dataset/llava/figures/easy_00000000.jpg",
        },
        {
            "prompt": "How many line segments are there in the image?\n<image>",
            "image": "dataset/llava/figures/easy_00000001.jpg",
        },
    ]
    outputs = vllm_llava(inputs)
    print(outputs)


if __name__ == "__main__":
    main()

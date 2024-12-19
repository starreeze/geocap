# -*- coding: utf-8 -*-
# @Date    : 2024-06-03 15:48:48
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator

import requests
import torch
import yaml
from transformers import pipeline

from common.args import logger, run_args
from common.iterwrap import IterateWrapper, retry_dec


class LLMGenerator(ABC):
    def __init__(self, *args, **kwargs) -> None:
        if "max_length" not in kwargs:
            self.kwargs = {"max_new_tokens": 1024} | kwargs

    @abstractmethod
    def __call__(
        self, input_texts: list[list[dict[str, str]]], batch_size=1, output_path: str | None = None
    ) -> Generator[list[str], Any, None]:
        pass


class LocalGenerator(LLMGenerator):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.generator = pipeline(
            "text-generation", model=model, device_map="auto", torch_dtype=torch.float16, **self.kwargs  # type: ignore
        )
        self.generator.tokenizer.padding_side = "left"  # type: ignore

    def __call__(self, input_texts: list[list[dict[str, str]]], batch_size=1, output_path: str | None = None):
        out_file = open(output_path, "w") if output_path else None
        target_range = range((len(input_texts) + batch_size - 1) // batch_size)
        if output_path:
            target_range = IterateWrapper(target_range, run_name="generate")
        for i in target_range:
            results: list[str] = []
            batch = input_texts[i * batch_size : (i + 1) * batch_size]
            outputs = self.generator(batch, batch_size=batch_size)
            for input_text, output in zip(batch, outputs):  # type: ignore
                generated_text = output[0]["generated_text"][len(input_text)]["content"].strip()
                results.append(generated_text)
                if out_file:
                    out_file.write(generated_text + "\n")
            yield results
        if out_file:
            out_file.close()


class Llama3Generator(LocalGenerator):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id[0]  # type: ignore


class APIGenerator(LLMGenerator):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model
        key_info: dict[str, str] = yaml.safe_load(open(run_args.api_key_file))
        self.url = (key_info["base_url"] + "/chat/completions").replace("//", "/")
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key_info['api_key']}"}

    @retry_dec(retry=10, wait=10)
    def get_one_response(self, text_inputs: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": text_inputs,
            "max_tokens": self.kwargs["max_new_tokens"],
        }
        raw = requests.post(self.url, headers=self.headers, json=payload)
        if raw.status_code != 200:
            raise Exception(f"Error calling api: {raw.status_code} {raw.text}")
        response = raw.json()
        logger.debug(f"Response from api: {response}")
        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        error = response["error"]
        if "inner_error" in error or "innererror" in error:
            logger.warning(error)
            return ""
        if error.get("code", "") == "429":
            raise Exception("API rate limit exceeded")
        raise Exception(f"Error calling api")

    def __call__(self, input_texts: list[list[dict[str, str]]], batch_size=1, output_path: str | None = None):
        if batch_size != 1:
            logger.warning("Batch size > 1 is not supported for API generator; setting batch_size to 1")
        out_file = open(output_path, "w") if output_path else None
        target_range = range(len(input_texts))
        if output_path:
            target_range = IterateWrapper(target_range, run_name="generate")
        for i in target_range:
            outputs = self.get_one_response(input_texts[i])
            if out_file:
                out_file.write(outputs + "\n")
            yield [outputs]
        if out_file:
            out_file.close()


generator_mapping: dict[str, type[LLMGenerator]] = {
    "llama31": Llama3Generator,
    "qwen25": LocalGenerator,
    "api": APIGenerator,
}
model_path_mapping = {
    "llama31": "/home/nfs02/model/llama-3.1-{}b-instruct",
    "qwen25": "/home/nfs02/model/Qwen2.5/Qwen2.5-{}B-Instruct",
    "api": "{}",
}


def main():
    messages = [[{"role": "user", "content": "Write a story beginning with 'Once upon a time'."}]]
    print(APIGenerator("api-gpt-4o")(messages))


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
# @Date    : 2024-06-03 15:48:48
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import torch
from transformers import pipeline
from tqdm import tqdm
from typing import cast


class LLMGenerator:
    def __init__(self, model: str, **kwargs) -> None:
        if "max_length" not in kwargs:
            kwargs = {"max_new_tokens": 1024} | kwargs
        self.generator = pipeline(
            "text-generation", model=model, device_map="auto", torch_dtype=torch.float16, **kwargs  # type: ignore
        )
        self.generator.tokenizer.padding_side = "left"  # type: ignore

    def __call__(self, input_texts, batch_size=1) -> list[str]:
        results = []
        for i in tqdm(range((len(input_texts) + batch_size - 1) // batch_size)):
            batch = input_texts[i * batch_size : (i + 1) * batch_size]
            outputs = self.generator(batch, batch_size=batch_size)
            for input_text, output in zip(batch, outputs):  # type: ignore
                generated_text = output[0]["generated_text"][len(input_text) :][0]["content"].strip()
                results.append(generated_text)
        return results


class Llama3Generator(LLMGenerator):
    def __init__(self, model: str, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id[0]  # type: ignore


generator_mapping: dict[str, type[LLMGenerator]] = {"llama3": Llama3Generator, "qwen2": LLMGenerator}
model_path_mapping = {
    "llama3": "/home/nfs02/model/llama-3.1-{size}b-instruct",
    "qwen2": "/home/nfs02/model/Qwen/Qwen2-{size}B-Instruct",
}


def main():
    print(LLMGenerator("meta/llama3-8b-chat")(["Once upon a time, "]))


if __name__ == "__main__":
    main()

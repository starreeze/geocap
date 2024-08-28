# -*- coding: utf-8 -*-
# @Date    : 2024-06-03 15:48:48
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations

import torch
from transformers import pipeline

from common.iterwrap import IterateWrapper


class LLMGenerator:
    def __init__(self, model: str, **kwargs) -> None:
        if "max_length" not in kwargs:
            kwargs = {"max_new_tokens": 1024} | kwargs
        self.generator = pipeline(
            "text-generation", model=model, device_map="auto", torch_dtype=torch.float16, **kwargs  # type: ignore
        )
        self.generator.tokenizer.padding_side = "left"  # type: ignore

    def __call__(self, input_texts, batch_size=1, output_path: str | None = None):
        out_file = open(output_path, "w") if output_path else None
        target_range = range((len(input_texts) + batch_size - 1) // batch_size)
        if output_path:
            target_range = IterateWrapper(target_range, run_name="generate")
        for i in target_range:
            results: list[str] = []
            batch = input_texts[i * batch_size : (i + 1) * batch_size]
            outputs = self.generator(batch, batch_size=batch_size)
            for input_text, output in zip(batch, outputs):  # type: ignore
                generated_text = output[0]["generated_text"][len(input_text) :][0]["content"].strip()
                results.append(generated_text)
                if out_file:
                    out_file.write(generated_text + "\n")
            yield results
        if out_file:
            out_file.close()


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

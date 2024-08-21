from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import cast
import torch
from transformers import pipeline
from tqdm import tqdm


class LLMGenerator:
    def __init__(self, model, **kwargs) -> None:
        self.generator = pipeline(
            "text-generation", model=model, device_map="auto", torch_dtype=torch.bfloat16, max_length=1024, **kwargs
        )
        self.generator.tokenizer.pad_token_id = self.generator.model.config.eos_token_id  # type: ignore
        self.generator.tokenizer.padding_side = "left"  # type: ignore

    def __call__(self, input_texts: list, batch_size=1) -> list[str]:
        results = []
        for i in tqdm(range((len(input_texts) + batch_size - 1) // batch_size)):
            batch = input_texts[i * batch_size : (i + 1) * batch_size]
            outputs = self.generator(batch, batch_size=batch_size)
            for input_text, output in zip(batch, outputs):  # type: ignore
                print(output)
                generated_text = output[0]["generated_text"][-1]["content"].strip()
                results.append(generated_text)
        return results


class Messages(BaseModel):
    messages: list


app = FastAPI()
gen = LLMGenerator("/home/nfs02/model/llama-3.1-70b-instruct")
# gen = print

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
def chat(msg: Messages):
    return cast(list, gen(msg.messages))[0]


def main():
    uvicorn.run(app=app, host="127.0.0.1", port=7999)


if __name__ == "__main__":
    main()

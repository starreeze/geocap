import json, requests

addr = "http://10.107.254.250:31023/llm/generate"


def generate(input: str, max_new_tokens: int = 2048, **model_params):
    payload = {
        "input": input,
        "serviceParams": {"promptTemplateName": "qwen2", "stream": False, "maxOutputLength": max_new_tokens},
        "history": [],
        "modelParams": model_params,
    }
    response = requests.post(addr, headers={"Content-Type": "application/json"}, json=payload)
    return json.loads(response.text)["output"]


if __name__ == "__main__":
    print(generate("Who are you?"))

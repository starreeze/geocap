from transformers import AutoModelForCausalLM, AutoTokenizer


def get_qwen_answer(question):
    print("load model")
    model = AutoModelForCausalLM.from_pretrained(
        "./model/Qwen2.5-14B-Instruct", torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("./model/Qwen2.5-14B-Instruct")
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

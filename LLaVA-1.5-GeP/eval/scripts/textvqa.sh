#!/bin/bash


python -m llava.eval.model_vqa_loader \
    --model-base /model/vicuna-7b-v1.5 \
    --model-path /model/llava-7b-lora\
    --question-file /playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /playground/data/eval/textvqa/train_images \
    --answers-file /LLaVA-1.5-GeP/eval/result/textvqa/answers/llava-v1.5-7b-lora.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file /playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /LLaVA-1.5-GeP/eval/result/textvqa/answers/llava-v1.5-7b-lora.jsonl

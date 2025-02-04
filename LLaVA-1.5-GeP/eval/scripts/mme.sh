#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-base /model/vicuna-7b-v1.5 \
    --model-path /model/llava-7b-lora\
    --question-file /playground/data/eval/MME/llava_mme_1.jsonl \
    --image-folder /playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /LLaVA-1.5-GeP/eval/result/MME/answers/llava-v1.5-7b-lora-bf16.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /LLaVA-1.5-GeP/eval/result/MME

python convert_answer_to_mme.py --experiment llava-v1.5-7b-lora-bf16

cd /playground/data/eval/MME/eval_tool

python calculation.py --results_dir /LLaVA-1.5-GeP/eval/result/MME/MME_Benchmark_release_version/eval_tool/answers/llava-v1.5-7b-lora-bf16

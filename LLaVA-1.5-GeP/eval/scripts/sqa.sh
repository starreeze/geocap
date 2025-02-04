#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-base /model/vicuna-7b-v1.5 \
    --model-path /model/llava-7b-lora\
    --question-file /playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /playground/data/eval/scienceqa/ScienceQA_DATA/test \
    --answers-file /LLaVA-1.5-GeP/eval/result/scienceqa/answers/llava-v1.5-7b-lora.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /LLaVA/playground/data/eval/scienceqa/ScienceQA_DATA \
    --result-file /LLaVA-1.5-GeP/eval/result/scienceqa/answers/llava-v1.5-7b-lora.jsonl \
    --output-file /LLaVA-1.5-GeP/eval/result/scienceqa/answers/llava-v1.5-7b-lora_output.jsonl \
    --output-result /LLaVA-1.5-GeP/eval/result/scienceqa/answers/llava-v1.5-7b-lora_result.json

#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-base /model/vicuna-7b-v1.5 \
    --model-path /model/llava-7b-lora\
    --question-file /playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file  /LLaVA-1.5-GeP/eval/result/mmbench/answers/$SPLIT/llava-v1.5-7b-lora-bf16.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p  /LLaVA-1.5-GeP/eval/result/mmbench/answers_upload/$SPLIT

python /LLaVA-1.5-GeP/convert_mmbench_for_submission.py \
    --annotation-file /playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir /LLaVA-1.5-GeP/eval/result/mmbench/answers/$SPLIT \
    --upload-dir /LLaVA-1.5-GeP/eval/result/mmbench/answers_upload/mmbench_dev_20230712 \
    --experiment llava-v1.5-7b-lora-bf16
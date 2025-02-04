#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-base /model/vicuna-7b-v1.5 \
    --model-path /model/llava-7b-lora\
    --question-file /playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /playground/data/eval/pope/val2014 \
    --answers-file /LLaVA-1.5-GeP/eval/result/pope/answers/llava-v1.5-7b-lora.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /playground/data/eval/pope/coco \
    --question-file /playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file /LLaVA-1.5-GeP/eval/result/pope/answers/llava-v1.5-7b-lora.jsonl

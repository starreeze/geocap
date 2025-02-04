#!/bin/bash

python -m llava.eval.model_vqa \
    --model-base /model/vicuna-7b-v1.5 \
    --model-path /model/llava-7b-lora\
    --question-file /playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /playground/data/eval/mm-vet/mm-vet/images \
    --answers-file /LLaVA-1.5-GeP/eval/result/mm-vet/answers/llava-v1.5-7b-lora.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /LLaVA-1.5-GeP/eval/result/mm-vet/results

python /LLaVA-1.5-GeP/eval/scripts/convert_mmvet_for_eval.py \
    --src /LLaVA-1.5-GeP/eval/result/mm-vet/answers/llava-v1.5-7b-lora.jsonl \
    --dst /LLaVA-1.5-GeP/eval/result/mm-vet/results/llava-v1.5-7b-lora.json


#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-base /model/vicuna-7b-v1.5 \
    --model-path /model/llava-7b-lora\
    --question-file /playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /playground/data/eval/vizwiz/test \
    --answers-file /LLaVA-1.5-GeP/eval/result/vizwiz/answers/llava-v1.5-7b-lora.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python  /LLaVA-1.5-GeP/eval/scripts/convert_vizwiz_for_submission.py \
    --annotation-file /playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file /LLaVA-1.5-GeP/eval/result/vizwiz/answers/llava-v1.5-7b-lora.jsonl \
    --result-upload-file /LLaVA-1.5-GeP/eval/result/vizwiz/answers_upload_1/llava-v1.5-viz-7b-lora.json

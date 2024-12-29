#!/bin/bash

# python -m llava.eval.model_vqa \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
#     --image-folder ./playground/data/eval/mm-vet/images \
#     --answers-file ./playground/data/eval/mm-vet/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p ./playground/data/eval/mm-vet/results

# python scripts/convert_mmvet_for_eval.py \
#     --src ./playground/data/eval/mm-vet/answers/llava-v1.5-13b.jsonl \
#     --dst ./playground/data/eval/mm-vet/results/llava-v1.5-13b.json

python -m llava.eval.model_vqa \
    --model-path checkpoints/finetune \
    --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/mm-vet/mm-vet/images \
    --answers-file eval/mmvet-l6-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

# mkdir -p ./playground/data/eval/mm-vet/results

python /home/nfs03/zhaof/LLaVA/scripts/convert_mmvet_for_eval.py \
    --src eval/mmvet-l6-7b.jsonl \
    --dst eval/mmvet-l6-7b.json

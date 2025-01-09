#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/geocap-s2-7b \
    --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/pope/val2014 \
    --answers-file eval/pope/l6-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir /home/nfs03/zhaof/LLaVA/playground/data/eval/pope/coco \
    --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file eval/pope/l6-7b.jsonl

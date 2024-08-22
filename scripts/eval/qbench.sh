#!/bin/bash

if [ "$1" = "dev" ]; then
    echo "Evaluating in 'dev' split."
elif [ "$1" = "test" ]; then
    echo "Evaluating in 'test' split."
else
    echo "Unknown split, please choose between 'dev' and 'test'."
    exit 1
fi

python -m llava.eval.model_vqa_qbench \
    --model-path checkpoints/geocap-s2-7b \
    --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/qbench/images/ \
    --questions-file /home/nfs03/zhaof/LLaVA/playground/data/eval/qbench/llvisionqa_$1.json \
    --answers-file eval/qbench/l6-7b.jsonl \
    --conv-mode llava_v1 \
    --lang en

python /home/nfs03/zhaof/LLaVA/playground/data/eval/qbench/format_qbench.py \
    --filepath eval/qbench/l6-7b.jsonl

python llava/eval/qbench_eval.py \
    --filepath eval/qbench/l6-7b.jsonl

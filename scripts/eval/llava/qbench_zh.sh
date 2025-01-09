#!/bin/bash

if [ "$1" = "dev" ]; then
    ZH_SPLIT="验证集"
    echo "Evaluating in 'dev' split."
elif [ "$1" = "test" ]; then
    ZH_SPLIT="测试集"
    echo "Evaluating in 'test' split."
else
    echo "Unknown split, please choose between 'dev' and 'test'."
    exit 1
fi

python -m llava.eval.model_vqa_qbench \
    --model-path checkpoints/finetune \
    --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/qbench_zh/images/ \
    --questions-file /home/nfs03/zhaof/LLaVA/playground/data/eval/qbench_zh/质衡-问答-$ZH_SPLIT.json \
    --answers-file eval/qbench-zh/l6-7b.jsonl \
    --conv-mode llava_v1 \
    --lang zh

python /home/nfs03/zhaof/LLaVA/playground/data/eval/qbench_zh/format_qbench.py \
    --filepath eval/qbench-zh/l6-7b.jsonl

python llava/eval/qbench_eval.py \
    --filepath eval/qbench-zh/l6-7b.jsonl

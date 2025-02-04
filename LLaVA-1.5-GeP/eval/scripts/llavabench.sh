#!/bin/bash

python -m llava.eval.model_vqa \
    --model-base /model/vicuna-7b-v1.5 \
    --model-path /model/llava-1.5-7b-lora\
    --question-file /playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder /playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file /LLaVA-1.5-GeP/eval/result/llava-bench-in-the-wild/answers/llava-v1.5-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p /LLaVA-1.5-GeP/eval/result/llava-bench-in-the-wild/reviews

python /LLaVA-1.5-GeP/eval/scripts/eval_qwen_review_bench.py \
    --question /playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context /playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule /LLaVA/llava/eval/table/rule.json \
    --answer-list \
        /playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        /LLaVA-1.5-GeP/eval/result/llava-bench-in-the-wild/answers/llava-v1.5-7b.jsonl \
    --output \
       /LLaVA-1.5-GeP/eval/result/llava-bench-in-the-wild/reviews/llava-v1.5-7b.jsonl\
    --model-path /model/Qwen2.5-14B-Instruct
python /LLaVA-1.5-GeP/eval/scripts/summarize_qwen_review.py -f /LLaVA-1.5-GeP/eval/result/llava-bench-in-the-wild/reviews/llava-v1.5-7b.jsonl

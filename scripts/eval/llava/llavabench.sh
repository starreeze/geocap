#!/bin/bash

# python -m llava.eval.model_vqa \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
#     --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

# python llava/eval/eval_gpt_review_bench.py \
#     --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
#     --rule llava/eval/table/rule.json \
#     --answer-list \
#         playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
#         playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b.jsonl \
#     --output \
#         playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl

# python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b.jsonl

# python -m llava.eval.model_vqa \
#     --model-path /home/nfs03/zhaof/LLaVA/playground/model/llava-v1.5-13b \
#     --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/images \
#     --answers-file /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b-1.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/reviews

python /home/nfs03/zhaof/LLaVA/llava/eval/eval_gpt_review_bench.py \
    --question /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule /home/nfs03/zhaof/LLaVA/llava/eval/table/rule.json \
    --answer-list \
        /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/answers/llava-v1.5-13b-1.jsonl \
    --output \
        /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b-1.jsonl

python /home/nfs03/zhaof/LLaVA/llava/eval/summarize_gpt_review.py -f /home/nfs03/zhaof/LLaVA/playground/data/eval/llava-bench-in-the-wild/reviews/llava-v1.5-13b-1.jsonl

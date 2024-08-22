#!/bin/bash

# python -m llava.eval.model_vqa_science \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder ./playground/data/eval/scienceqa/images/test \
#     --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
#     --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
#     --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json

python -m llava.eval.model_vqa_science \
    --model-path checkpoints/geocap-s2-fe \
    --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/scienceqa/ScienceQA_DATA/test \
    --answers-file eval/sqa/l6-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /home/nfs03/zhaof/LLaVA/playground/data/eval/scienceqa/ScienceQA_DATA \
    --result-file eval/sqa/l6-7b.jsonl \
    --output-file eval/sqa/l6-7b-output.jsonl \
    --output-result eval/sqa/l6-7b-result.json

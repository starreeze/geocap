#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path /home/nfs03/zhaof/LLaVA/playground/model/llava-v1.5-13b \
#     --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/llava_mme.jsonl \
#     --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
#     --answers-file /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# cd /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/

# python convert_answer_to_mme.py --experiment llava-v1.5-13b

# cd eval_tool

# python calculation.py --results_dir answers/llava-v1.5-13b

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/llava-s2 \
    --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/llava_mme_1.jsonl \
    --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/answers/l6-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /home/nfs03/zhaof/LLaVA/playground/data/eval/MME

python convert_answer_to_mme.py --experiment l6-7b

cd eval_tool

python calculation.py --results_dir /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version/eval_tool/answers/l6-7b

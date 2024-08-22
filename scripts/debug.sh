#!/bin/bash

python -m pdb -m llava.eval.model_vqa_loader \
    --model-path checkpoints/geocap-s2-7b \
    --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/llava_mme_1.jsonl \
    --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file /home/nfs03/zhaof/LLaVA/playground/data/eval/MME/answers/l6-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

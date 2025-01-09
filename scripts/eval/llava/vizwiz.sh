#!/bin/bash

# python -m llava.eval.model_vqa_loader \
#     --model-path /home/nfs03/zhaof/LLaVA/playground/model/llava-v1.5-13b \
#     --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
#     --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/test \
#     --answers-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python /home/nfs03/zhaof/LLaVA/scripts/convert_vizwiz_for_submission.py \
#     --annotation-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
#     --result-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/answers/llava-v1.5-13b.jsonl \
#     --result-upload-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/answers_upload/llava-v1.5-13b.json


python -m llava.eval.model_vqa_loader \
    --model-path /home/nfs03/zhaof/LLaVA/playground/model/llava-v1.5-13b \
    --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/test \
    --answers-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/answers/llava-v1.5-13b-1.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python /home/nfs03/zhaof/LLaVA/scripts/convert_vizwiz_for_submission.py \
    --annotation-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/answers/llava-v1.5-13b-1.jsonl \
    --result-upload-file /home/nfs03/zhaof/LLaVA/playground/data/eval/vizwiz/answers_upload_1/llava-v1.5-13b.json

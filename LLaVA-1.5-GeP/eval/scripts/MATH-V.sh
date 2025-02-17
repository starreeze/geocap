#!/bin/bash
export PYTHONPATH=`pwd`
python /MATH-V/evaluation/get_response.py \
    --model_path /model/llava-1.5-7b\
    --test_file /MATH-V/data/test.jsonl\
    --image_dir /MATH-V/data\
    --response_file MATH-V/data/llava-1.5-7b-response.json

python /MATH-V/evaluation/evaluate.py\
    --response_file /MATH-V/data/llava-1.5-7b-response.json\

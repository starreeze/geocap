#!/bin/bash
export PYTHONPATH=`pwd`
python /MathVista/get_response.py \
    --model_path /model/llava-1.5-7b\
    --query_file /MathVista/data/query.json\
    --test_file  /MathVista/data/testmini.json \
    --image_dir /MathVista/data/\
    --response_file /MathVista/data/llava-1.5-7b-response.json

python /MathVista/extract_answer.py \
    --query_file /MathVista/data/query.json\
    --test_file  /MathVista/data/testmini.json \
    --response_file /MathVista/data/llava-1.5-7b-response.json\
    --model_path /model/Qwen2.5-14B-Instruct\
    --output_file /MathVista/data/llava-1.5-7b-answer.json

python /MathVista/score.py  \
    --groundtruth_file  /MathVista/data/testmini.json \
    --output_file /MathVista/data/llava-1.5-7b-answer.json\
    --score_file /MathVista/data/llava-1.5-7b-score.json
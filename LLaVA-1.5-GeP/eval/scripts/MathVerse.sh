#!/bin/bash
export PYTHONPATH=`pwd`
python /MathVerse/evaluation/get_response.py \
    --model_path /model/llava-1.5-7b \
    --test_file /MathVerse/data/testmini.json\
    --image_dir  /MathVerse/data/images \
    --response_file /MathVerse/data/llava-1.5-7b-response.json

python /MathVerse/evaluation/extract_answer.py \
    --model_output_file /MathVerse/data/llava-1.5-7b-response.json \
    --save_file /MathVerse/data/llava-1.5-7b-answer.json\
    --model_path /model/Qwen2.5-14B-Instruct

python /MathVerse/evaluation/score.py \
    --answer_extraction_file /MathVerse/data/llava-1.5-7b-answer.json \
    --save_file /MathVerse/data/llava-1.5-7b-score.json
#!/bin/bash
export PYTHONPATH=`pwd`

python ChartQA/ChartQADataset/test/get_response.py\
    --model_path /model/llava-1.5-7b\
    --test_file ChartQA/ChartQADataset/test/test_augmented.json\
    --image_dir ChartQA/ChartQADataset/test/png\
    --response_file ChartQA/ChartQADataset/test/results/llava-1.5-7b-augmented-response.json

python ChartQA/ChartQADataset/test/get_response.py\
    --model_path /model/llava-1.5-7b\
    --test_file ChartQA/ChartQADataset/test/test_human.json\
    --image_dir ChartQA/ChartQADataset/test/png\
    --response_file ChartQA/ChartQADataset/test/results/llava-1.5-7b-human-response.json

python ChartQA/ChartQADataset/test/extract_answer.py\
    --response_file ChartQA/ChartQADataset/test/results/llava-1.5-7b-augmented-response.json\
    --model_path /model/Qwen2.5-14B-Instruct\
    --output_file ChartQA/ChartQADataset/test/results/llava-1.5-7b-augmented-answer.json

python ChartQA/ChartQADataset/test/extract_answer.py\
    --response_file ChartQA/ChartQADataset/test/results/llava-1.5-7b-human-response.json\
    --model_path /model/Qwen2.5-14B-Instruct\
    --output_file ChartQA/ChartQADataset/test/results/llava-1.5-7b-human-answer.json

python ChartQA/ChartQADataset/test/score.py\
    --input_file ChartQA/ChartQADataset/test/results/llava-1.5-7b-augmented-answer.json

python ChartQA/ChartQADataset/test/score.py\
    --input_file ChartQA/ChartQADataset/test/results/llava-1.5-7b-human-answer.json




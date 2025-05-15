#!/bin/bash
export PYTHONPATH=`pwd`
python ./LLaVA-1.5-GeP/eval/PMC-VQA/test.py\
    --csv_path ./PMC-VQA/test.csv \
    --img_dir ./PMC-VQA/figures/ \
    --model_path ./model/llava-1.5-7b \
    --model_name llava-1.5-7b\
    --output_dir ./PMC-VQA/result

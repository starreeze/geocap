#!/bin/bash
export PYTHONPATH=`pwd`
python CharXiv/src/generate.py \
    --data_dir CharXiv/data \
    --image_dir CharXiv/images\
    --output_dir CharXiv/results\
    --split val\
    --mode descriptive\
    --model_path /model/llava-1.5-7b\
    --model_name llava-1.5-7b

python CharXiv/src/generate.py \
    --data_dir CharXiv/data \
    --image_dir CharXiv/images\
    --output_dir /CharXiv/results\
    --split val\
    --mode reasoning\
    --model_path /model/llava-1.5-7b\
    --model_name llava-1.5-7b

python CharXiv/src/evaluate.py \
    --model_name llava-1.5-7b\
    --split val\
    --mode reasoning
python CharXiv/src/evaluate1.py \
    --model_name llava-1.5-7b\
    --split val\
    --mode descriptive
python CharXiv/src/get_stats.py \
    --model_name llava-1.5-7b 
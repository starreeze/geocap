#!/bin/bash
export PYTHONPATH=`pwd`
python ./LLaVA-1.5-GeP/eval/SLAKE/get_response.py\
    --input_file ./SLAKE/test.json \
    --img_dir ./SLAKE/imgs\
    --model_path ./model/llava-1.5-7b \
    --model_name llava-1.5-7b\
    --output_dir ./SLAKE/result

python ./LLaVA-1.5-GeP/eval/SLAKE/score.py\
    --input_file ./SLAKE/result/gen-llava-1.5-7b.json
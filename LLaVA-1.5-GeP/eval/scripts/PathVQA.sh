#!/bin/bash
export PYTHONPATH=`pwd`
python ./LLaVA-1.5-GeP/eval/Path-VQA/get_response.py\
    --input_files ./PathVQA/data/test-*.parquet \
    --model_path ./model/llava-1.5-7b \
    --output_dir ./PathVQA\
    --model_name llava-1.5-7b

python ./LLaVA-1.5-GeP/eval/SLAKE/score.py\
    --input_file ./PathVQA/result/gen-llava-1.5-7b.json
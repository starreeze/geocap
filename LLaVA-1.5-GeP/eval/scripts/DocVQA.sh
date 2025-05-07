#!/bin/bash
export PYTHONPATH=`pwd`
python /home/nfs05/xingsy/wzt/geocap-1/geocap/LLaVA-1.5-GeP/eval/Docvqa/get_response.py\
    --model_path /home/nfs05/xingsy/wzt/geocap/model/llava-1.5-7b \
    --input_files /home/nfs05/xingsy/wzt/geocap/DocVQA/DocVQA/validation-*.parquet \
    --output_file /home/nfs05/xingsy/wzt/geocap/DocVQA/DocVQA/llava-1.5-7b.json

python ./LLaVA-1.5-GeP/eval/Docvqa/score.py\
    --input_file ./DocVQA/llava-1.5-7b.json
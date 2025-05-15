#!/bin/bash
export PYTHONPATH=`pwd`
python ./LLaVA-1.5-GeP/eval/MedXpertQA/test.py\
    --input_file ./MedXpertQA/MM/test.jsonl \
    --image_dir ./MedXpertQA/images \
    --output_dir ./MedXpertQA/result\
    --model_path ./model/llava-1.5-7b \
    --model_name llava-1.5-7b
    

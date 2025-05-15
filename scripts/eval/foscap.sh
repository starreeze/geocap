#!/bin/bash
reference_file=eval_data/ds_eval_v3/extracted_reference_info.json
origin_file=eval_data/origin_files/stage3_only_latest_8848.json
result_dir=eval_data/ds_eval_v3/stage3_only_latest_8848
eval_llm=api-deepseek-chat

python run \
  --module eval.eval \
  --read_extraction False \
  --eval_reference_file $reference_file \
  --eval_origin_file $origin_file \
  --eval_result_dir $result_dir \
  --eval_llm $eval_llm \
  --eval_batchsize 1

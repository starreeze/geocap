#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-7b-lora"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
      --model-base /model/vicuna-7b-v1.5 \
      --model-path /model/llava-7b-lora\
      --question-file /playground/data/eval/seed_bench/llava-seed-bench-2.jsonl \
      --image-folder /playground/data/eval/seed_bench/ \
      --answers-file /LLaVA-1.5-GeP/eval/result/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl \
      --num-chunks $CHUNKS \
      --chunk-idx $IDX \
      --temperature 0 \
      --conv-mode vicuna_v1 &
done

wait

# output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl
output_file=/LLaVA-1.5-GeP/eval/result/seed_bench/answers/$CKPT/merge_1.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
   cat /LLaVA-1.5-GeP/eval/result/seed_bench/answers/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python /LLaVA-1.5-GeP/eval/scripts/convert_seed_for_submission.py \
    --annotation-file /data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file /LLaVA-1.5-GeP/eval/result/seed_bench/answers_upload_1/llava-v1.5-7b-lora.jsonl


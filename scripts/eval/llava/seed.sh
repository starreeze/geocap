#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path checkpoints/finetune \
        --question-file /home/nfs03/zhaof/LLaVA/playground/data/eval/seed_bench/llava-seed-bench-2.jsonl \
        --image-folder /home/nfs03/zhaof/LLaVA/playground/data/eval/seed_bench/ \
        --answers-file eval/seed-l6-7b/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

# output_file=./playground/data/eval/seed_bench/answers/$CKPT/merge.jsonl
output_file=eval/seed-l6-7b/merge_1.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat eval/seed-l6-7b/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# Evaluate
python scripts/convert_seed_for_submission.py \
    --annotation-file /home/nfs03/zhaof/LLaVA/playground/data/eval/seed_bench/SEED-Bench.json \
    --result-file $output_file \
    --result-upload-file /home/nfs03/zhaof/LLaVA/playground/data/eval/seed_bench/answers_upload_1/llava-v1.5-13b-1.jsonl


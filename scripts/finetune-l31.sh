#!/bin/bash
export PYTHONPATH=`pwd`
#    --pretrain_visual_encoder None \

deepspeed llava/train/train_mem.py  \
    --model_name_or_path /home/nfs02/model/llama-3.1-8b-instruct \
    --deepspeed scripts/zero3_offload.json \
    --version v3 \
    --data_path /home/nfs03/zhaof/LLaVA/playground/data/llava_v1_5_mix665k.json \
    --image_folder /home/nfs03/zhaof/LLaVA/playground/data/ \
    --vision_tower /home/nfs03/zhaof/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter checkpoints/geocap-s1-fe/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_visual_encoder False \
    --tune_llm True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/geocap-s2-fe \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
export CUDA_VISIBLE_DEVICES=0 

MODEL_PATH=/home/nfs05/models/internvl_2.5/InternVL/internvl_chat/work_dirs/foscap/stage3_only_latest_6666

python run -m stage3.internvl.tag_decoding \
    --foscap_model_path $MODEL_PATH
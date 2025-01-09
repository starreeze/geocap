DATADIR=dataset/test/llava

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/gllava-s2 \
    --question-file $DATADIR/$1.jsonl \
    --image-folder dataset/test/figures \
    --answers-file $DATADIR/answers/$1.jsonl \
    --temperature 0.2 \
    --max_new_tokens 512 \
    --conv-mode vicuna_v1
DATADIR=dataset/test/llava

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/gllava-s2 \
    --question-file $DATADIR/$1.json \
    --image-folder $DATADIR/figures \
    --answers-file $DATADIR/answers/$1.json \
    --temperature 0 \
    --conv-mode vicuna_v1
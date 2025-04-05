#!/usr/bin/env bash

SPLIT=$1

if [ $SPLIT == "train" ]; then
    NUM_WORKERS=10
elif [ $SPLIT == "test" ]; then
    NUM_WORKERS=1
else
    echo "Invalid split: $SPLIT"
    exit 1
fi

python run.py --module data.draw.draw \
    --rules_path dataset/rules-easy.json --figure_dir dataset/figures \
    --figure_prefix easy --num_workers $NUM_WORKERS \
    --line_weight 2 --line_style none --randomize False

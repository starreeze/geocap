#!/usr/bin/env bash

SPLIT=$1

if [ $SPLIT == "train" ]; then
    NUM_WORKERS=100
elif [ $SPLIT == "test" ]; then
    NUM_WORKERS=10
else
    echo "Invalid split: $SPLIT"
    exit 1
fi

python run.py --module data.draw.draw \
    --rules_path dataset/rules-hard.json --figure_dir dataset/figures \
    --figure_prefix hard --num_workers $NUM_WORKERS --log_level warning \
    --Gaussian_proba 0.5 --Perlin_proba 0 --n_white_line 0 \
    --line_weight 3 --line_style xkcd

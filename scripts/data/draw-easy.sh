#!/usr/bin/env bash

python run.py --module data.draw.draw \
    --rules_path dataset/rules-easy.json --figure_dir dataset/figures \
    --figure_prefix easy --num_workers 100 \
    --line_weight 2 --line_style none --randomize False

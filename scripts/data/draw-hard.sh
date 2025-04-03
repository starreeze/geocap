#!/usr/bin/env bash

python run.py --module data.draw.draw \
    --rules_path dataset/rules-hard.json --figure_dir dataset/figures \
    --figure_prefix hard --num_workers 100 --log_level warning \
    --Gaussian_proba 0.5 --Perlin_proba 0 \
    --line_weight 3 --line_style xkcd

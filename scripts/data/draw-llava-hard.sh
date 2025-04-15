#!/usr/bin/env bash

./run -m data.draw.draw --line_weight 3 --line_style xkcd --rules_path dataset/llava/rules-hard.json --figure_dir dataset/llava/figures --figure_prefix hard --num_workers 100 --log_level error --Gaussian_proba 0.5 --Perlin_proba 0.5

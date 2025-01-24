#!/usr/bin/env bash

python run.py --module data.rule.generate --min_num_shapes 5 --num_samples 11401 10588 9785 8226 --in_canvas_area_thres 1 --polygon_shape_level 5 --line_shape_level 2 --ellipse_shape_level 3 --spiral_shape_level 1 --rules_path dataset/rules-hard.json --num_workers 64

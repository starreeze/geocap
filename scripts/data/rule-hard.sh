#!/usr/bin/env bash

python run.py --module data.rule.generate --max_num_shapes 8 --min_num_shapes 5 --in_canvas_area_thres 1 --polygon_shape_level 5 --line_shape_level 2 --ellipse_shape_level 3 --spiral_shape_level 1 --num_basic 40000 --rules_path dataset/rules-hard.json

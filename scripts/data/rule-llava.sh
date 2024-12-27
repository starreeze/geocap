#!/usr/bin/env bash
LEVEL=$1

python run.py --module data.rule.generate --max_num_shapes 8 --min_num_shapes 2 --in_canvas_area_thres 0.8 --polygon_shape_level 5 --line_shape_level 2 --ellipse_shape_level 3 --spiral_shape_level 1 --num_basic 150000 --rules_path dataset/llava/rules-$LEVEL.json

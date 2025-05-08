#!/usr/bin/env bash
LEVEL=$1

./run -m data.rule.generate --min_num_shapes 4 --num_samples_per_num_shapes 30000 30000 30000 30000 30000 \
    --rules_path dataset/llava/rules-$LEVEL.json --num_workers 64 \
    --polygon_shape_level 5 --line_shape_level 2 --ellipse_shape_level 3 --spiral_shape_level 1 \
    --polygon_symmetric_level 0 --polygon_shared_edge_level 0 --polygon_diagonal_level 0

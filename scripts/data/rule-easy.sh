#!/usr/bin/env bash

SPLIT=$1

if [ $SPLIT == "train" ]; then
    NUM_SAMPLES="13435 13049 13216"
    NUM_WORKERS=64
elif [ $SPLIT == "test" ]; then
    NUM_SAMPLES="100 100 100"
    NUM_WORKERS=4
else
    echo "Invalid split: $SPLIT"
    exit 1
fi

echo "Generating easy ruleset for $SPLIT set with [$NUM_SAMPLES] samples"

./run -m data.rule.generate --min_num_shapes 2 --num_samples $NUM_SAMPLES \
    --rules_path dataset/rules-easy.json --num_workers $NUM_WORKERS \
    --in_canvas_area_thres 1 \
    --polygon_shape_level 5 --line_shape_level 2 --ellipse_shape_level 3 --spiral_shape_level 1 \
    --polygon_tangent_line_level 0 --polygon_symmetric_level 0 --polygon_shared_edge_level 0 --polygon_diagonal_level 0 \
    --star_circumscribed_polygon_level 0

#!/usr/bin/env bash

SPLIT=$1

if [ $SPLIT == "train" ]; then
    NUM_SAMPLES="14321 10518 9705 8146"
    NUM_WORKERS=64
elif [ $SPLIT == "test" ]; then
    NUM_SAMPLES="80 80 80 80"
    NUM_WORKERS=4
else
    echo "Invalid split: $SPLIT"
    exit 1
fi

echo "Generating hard ruleset for $SPLIT set with [$NUM_SAMPLES] samples"

python run.py --module data.rule.generate --min_num_shapes 5 --num_samples $NUM_SAMPLES \
    --rules_path dataset/rules-hard.json --num_workers $NUM_WORKERS \
    --in_canvas_area_thres 1 \
    --polygon_shape_level 5 --line_shape_level 2 --ellipse_shape_level 3 --spiral_shape_level 1 \
    --polygon_tangent_line_level 0 --polygon_symmetric_level 0 --polygon_shared_edge_level 0 --polygon_diagonal_level 0

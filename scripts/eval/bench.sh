#!/usr/bin/env bash
LEVEL=$1
shift 1 # Shift positional parameters so $3 becomes $1, and so on
ARGS="$@"

python run.py --module eval.gepbench --vqa_question_dir dataset/vqa-${LEVEL} --vqa_output_dir results/${LEVEL} --figure_dir dataset/figures --figure_prefix ${LEVEL} ${ARGS}

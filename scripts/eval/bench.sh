#!/usr/bin/env bash
LEVEL=$1
shift 1
ARGS="$@"

./run -m eval.gepbench --vqa_question_dir dataset/vqa-${LEVEL} --vqa_output_dir results/${LEVEL} --figure_dir dataset/figures --figure_prefix ${LEVEL} ${ARGS}

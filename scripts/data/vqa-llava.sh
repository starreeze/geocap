#!/usr/bin/env bash
LEVEL=$1

./run -m data.vqa.question --rules_path dataset/llava/rules-$LEVEL.json --vqa_question_dir dataset/llava/vqa-$LEVEL

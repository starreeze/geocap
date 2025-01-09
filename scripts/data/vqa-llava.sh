#!/usr/bin/env bash
LEVEL=$1

python run.py --module data.vqa.question --rules_path dataset/llava/rules-$LEVEL.json --vqa_question_dir dataset/llava/vqa-$LEVEL

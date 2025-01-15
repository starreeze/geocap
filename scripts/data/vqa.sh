#!/usr/bin/env bash
LEVEL=$1

python run.py --module data.vqa.question --rules_path dataset/rules-$LEVEL.json --vqa_question_dir dataset/vqa-$LEVEL --end_pos 10000 --perspectives existence counting size reference

python run.py --module data.vqa.question --rules_path dataset/rules-$LEVEL.json --vqa_question_dir dataset/vqa-$LEVEL --end_pos 15000 --perspectives location

python run.py --module data.vqa.question --rules_path dataset/rules-$LEVEL.json --vqa_question_dir dataset/vqa-$LEVEL --end_pos 40000 --perspectives relation

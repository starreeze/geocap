#!/usr/bin/env bash

LEVEL=$1
SPLIT=$2

if [ $SPLIT == "train" ]; then
    NUM_SAMPLE="10000 15000 40000"
elif [ $SPLIT == "test" ]; then
    NUM_SAMPLE="70 110 300"
else
    echo "Invalid split: $SPLIT"
    exit 1
fi

read -r -a SAMPLES <<<"$NUM_SAMPLE"

python run.py --module data.vqa.question --rules_path dataset/rules-$LEVEL.json --vqa_question_dir dataset/vqa-$LEVEL --end_pos ${SAMPLES[0]} --perspectives existence counting size reference

python run.py --module data.vqa.question --rules_path dataset/rules-$LEVEL.json --vqa_question_dir dataset/vqa-$LEVEL --end_pos ${SAMPLES[1]} --perspectives location

python run.py --module data.vqa.question --rules_path dataset/rules-$LEVEL.json --vqa_question_dir dataset/vqa-$LEVEL --end_pos ${SAMPLES[2]} --perspectives relation

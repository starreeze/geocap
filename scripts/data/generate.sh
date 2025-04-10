#! /usr/bin/env bash

SPLIT=$1
if [ $SPLIT == "llava" ]; then
    for LEVEL in easy hard; do
        bash scripts/data/rule-llava.sh $LEVEL
        bash scripts/data/draw-llava-$LEVEL.sh
        bash scripts/data/vqa-llava.sh $LEVEL
    done
else
    for LEVEL in easy hard; do
        bash scripts/data/rule-$LEVEL.sh $SPLIT
        bash scripts/data/draw-$LEVEL.sh $SPLIT
        bash scripts/data/vqa.sh $LEVEL $SPLIT
    done
fi

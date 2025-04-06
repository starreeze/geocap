#! /usr/bin/env bash

SPLIT=$1

for LEVEL in easy hard; do
    bash scripts/data/rule-$LEVEL.sh $SPLIT
    bash scripts/data/draw-$LEVEL.sh $SPLIT
    bash scripts/data/vqa.sh $LEVEL $SPLIT
done

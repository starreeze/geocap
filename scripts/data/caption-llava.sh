#!/usr/bin/env bash
LEVEL=$1
shift 1

python run.py --module data.caption.caption --rules_path dataset/llava/rules-$LEVEL.json --caption_dir dataset/llava/captions-$LEVEL $@

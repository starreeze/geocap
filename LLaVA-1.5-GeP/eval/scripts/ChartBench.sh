#!/bin/bash
export PYTHONPATH=`pwd`

python ./ChartBench/Repos/LLaVA/infer.py
python ./ChartBench/Stat/gpt_filter.py
python ./ChartBench/Stat/stat_all_metric.py
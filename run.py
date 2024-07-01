# -*- coding: utf-8 -*-
# @Date    : 2024-04-23 20:41:59
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from __future__ import annotations
import importlib
from common.args import run_args


if __name__ == "__main__":
    task = importlib.import_module(run_args.module)
    task.main()

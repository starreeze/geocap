import data.pil_backend as pld
import data.plt_backend as ptd
from common.args import data_args, run_args
from common.iterwrap import iterate_wrapper
import json
import os
from typing import Any


def draw_figure(rules: "dict", path: str, backend: str = "plt", random_seed=None, randomize=True):
    if backend == "plt":
        figure = ptd.Figure(rules, random_seed, randomize, xkcd=True)
    elif backend == "pil":
        figure = pld.Figure(rules, random_seed, randomize)
    else:
        raise ValueError(f"{backend} is not a valid backend.")
    figure.draw(stylish=True)
    figure.save(path)


def process_single(f, idx_sample: tuple[int, dict], vars):
    draw_figure(idx_sample[1], os.path.join(data_args.figure_dir, f"{idx_sample[0]:08d}.jpg"))


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)
        assert isinstance(samples, list)
    """ Serial Implement, can't work on my pc but works @ remote
    for idx_sample, sample in enumerate(samples):
        draw_figure(sample, os.path.join(data_args.figure_dir, f"{idx_sample:08d}.jpg"))
    # """
    iterate_wrapper(process_single, list(enumerate(samples)), num_workers=8)


if __name__ == "__main__":
    main()

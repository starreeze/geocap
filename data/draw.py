import json
import os

import data.pil_backend as pld
import data.plt_backend as ptd
from common.args import data_args, draw_args, run_args
from common.iterwrap import iterate_wrapper


def draw_figure(rules: "dict", path: str, backend: str = "plt", random_seed=None, randomize=True):
    if backend == "plt":
        figure = ptd.Figure(
            rules,
            random_seed,
            randomize,
            xkcd=draw_args.xkcd,
            size=draw_args.size,
            dpi=draw_args.dpi,
            line_weight=draw_args.line_weight,
        )
    elif backend == "pil":
        figure = pld.Figure(
            rules,
            random_seed,
            randomize,
            size=(
                int(draw_args.dpi * draw_args.size[0]),
                int(draw_args.dpi * draw_args.size[1]),
            ),
            line_weight=draw_args.line_weight,
        )
    else:
        raise ValueError(f"{backend} is not a valid backend.")
    figure.draw(
        color=draw_args.color,
        n_white_line=draw_args.n_white_line,
        Gaussian_mean=draw_args.Gaussian_mean,
        Gaussian_var=draw_args.Gaussian_var,
        Perlin_lattice=draw_args.Perlin_lattice,
        Perlin_bias=draw_args.Perlin_bias,
        Perlin_power=draw_args.Perlin_power,
        stylish=draw_args.stylish,
    )
    figure.save_release(path)


def process_single(f, idx_sample: tuple[int, dict], vars):
    draw_figure(
        idx_sample[1],
        os.path.join(
            data_args.figure_dir, data_args.figure_name.format(prefix=data_args.figure_prefix, id=idx_sample[0])
        ),
        draw_args.backend,
        draw_args.random_seed,
        draw_args.randomize,
    )


def main():
    with open(data_args.rules_path, "r") as f:
        samples = json.load(f)[run_args.start_pos : run_args.end_pos]
        assert isinstance(samples, list)
    serial_version = draw_args.serial_version
    if serial_version:
        for idx, sample in enumerate(samples):
            draw_figure(
                sample,
                os.path.join(
                    data_args.figure_dir, data_args.figure_name.format(prefix=data_args.figure_prefix, id=idx)
                ),
                draw_args.backend,
                draw_args.random_seed,
                draw_args.randomize,
            )
    else:
        iterate_wrapper(
            process_single,
            list(enumerate(samples)),
            num_workers=run_args.num_workers,
            run_name="draw",
            bar=run_args.progress_bar,
        )


if __name__ == "__main__":
    main()

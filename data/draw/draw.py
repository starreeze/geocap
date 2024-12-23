import json
import os
from tqdm import tqdm
import data.draw.pil_backend as pld
import data.draw.plt_backend as ptd
from common.args import data_args, draw_args, run_args
from common.iterwrap import iterate_wrapper
from typing import cast


def draw_figure(rules: "dict", path: str, backend: str = "plt", random_seed=None, randomize=True):
    # Color-Safe Check
    if draw_args.color == []:
        color = None
    else:
        assert len(draw_args.color) == 3
        color = draw_args.color

    assert len(draw_args.size) == 2

    if draw_args.line_style == "none":
        xkcd = False
        gradient = False
    elif draw_args.line_style == "gradient":
        xkcd = False
        gradient = True
    elif draw_args.line_style == "xkcd":
        xkcd = True
        gradient = False
    else:
        raise ValueError("Invalid line style, not any of ['none', 'gradient', 'xkcd']")

    if backend == "plt":
        figure = ptd.Figure(
            rules,
            random_seed,
            randomize,
            xkcd=xkcd,
            gradient=gradient,
            size=cast(tuple[float, float], tuple(draw_args.size)),
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
    if not randomize and color is None:
        color = [0, 0, 0]
    figure.draw(
        color=color,
        n_white_line=draw_args.n_white_line,
        white_line_radius=draw_args.white_line_range,
        Gaussian_mean=draw_args.Gaussian_mean,
        Gaussian_var=draw_args.Gaussian_var,
        Perlin_lattice=draw_args.Perlin_lattice,
        Perlin_bias=draw_args.Perlin_bias,
        Perlin_power=draw_args.Perlin_power,
        stylish=draw_args.stylish,
        proba=draw_args.proba,
    )
    figure.save_release(path)


def process_single(f, idx_sample: tuple[int, dict], vars):
    draw_figure(
        idx_sample[1],
        os.path.join(
            data_args.figure_dir,
            data_args.figure_name.format(prefix=data_args.figure_prefix, id=idx_sample[0]),
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
    os.makedirs(data_args.figure_dir, exist_ok=True)

    if serial_version:
        for idx, sample in tqdm(enumerate(samples), total=len(samples)):
            draw_figure(
                sample,
                os.path.join(
                    data_args.figure_dir,
                    data_args.figure_name.format(prefix=data_args.figure_prefix, id=idx),
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

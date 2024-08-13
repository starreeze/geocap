# GeoCap

geometry caption & geography fossil caption

## Install

Requirements are provided in `deploy/requirements.txt`. It's recommended to use python 3.10.

## Running a module

A `run.py` is provided for running a module. This is an elegant workaround for importing errors from different packages. Examples:

```shell
python run.py --module data.rules --num_basic_geo_samples 10  # default entry is main()
python run.py --module data.format --action to_llava  # you can also specify the entry function (--action)
```

### Running Module 'draw'

Two python files, `draw_PIL.py` and `draw_plt.py` is provided, in which the former one is written in pillow, providing continuous change of shape, and a relatively less noisy image; the latter, in comparison, provides hand-drawing line style and more natural noise. `draw_plt.py` is recommended to use and `run.py` will automatically parse this version.

To use `draw_plt.py`, the following arguments are expected:
- rules: "list[dict[str, Any]]". Mandatory. The rules you would like to draw.
- random_seed: int|None. The default value is None. Control the random seed.
- randomize: bool. The default value is True. Enable the noise-applying procedure.
- size: "tuple[float, float]". The deault value is (6.4, 6.4).
- dpi: int. The default value is 100. dpi * size = resolution.
- line_weight: int. The default value is 4. Control the line weight. If `randomize` is enabled, the line weight will be randomly chosen in a certain range near the value.
- xkcd: bool. The default value is False. Enable the hand-drawing line style.
- color:None|tuple[int,int,int]. The default value is None. If a color in RGB form is provided, that rule will be drawn in the given color. The the value is None, that rule will be drawn in random colors.
- n_white_line:None|int. The default value is None. If an integer is given, the white lines will be drawn in that certain amount. Otherwise, the value is randomly chosen.
- Gaussian_mean: float. The default value is 0. Control the mean value of the Gaussian noise. The higher the value is, the grayer the image will be.
- Gaussian_var: float. The default value is 10. Control the variance of the Gaussian Noise. The higher the value is, the stronger the Gaussian Noise will be.
- Perlin_lattice: int. The default value is 20. Control the number of lattices while generating Perlin noise. The value is not recommended to change and may cause the crash the the module.
- Perlin_power: float. The default value is 16. Control the power of the Perlin noise, will affect the contrast ratio of the noise and the image.
- Perlin_bias: float. The default value is -16. Control the bias of the Perlin noise. The lower it is, the brighter the image will be.
- stylish: bool. The default value is False. Setting to true will sharpen the image.

If you would like to use another version, please edit `args.py` and refer to `draw_PIL.py`. Most arguments are the same.

## Contributing

Fork and open a pull request. Follow the instructions below or your PR will fail.

1. Use `Pylance` (basic level) to lint your code while doing your work. Refer to https://docs.pydantic.dev/latest/integrations/visual_studio_code/#configure-vs-code to configure your VSCode. NOTE: Be cautious of using `# type: ignore` to suppress type errors, as you may be ignoring valuable traces of bugs; usually typing.cast() is more preferred.
2. Use `black` to format your code before opening a PR:

   ```shell
   pip install black
   black . --line-length 120 --extend-exclude models
   ```

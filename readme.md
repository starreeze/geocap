# GeoCap

geometry caption & geography fossil caption

## Install

Requirements are provided in `deploy/requirements.txt`. It's recommended to use python 3.10.

## Running a module

A `run.py` is provided for running a module. This is an elegant workaround for importing errors from different packages. Examples:

```shell
python run.py --module data.rule.generate --num_basic_geo_samples 10  # default entry is main()
python run.py --module data.format --action to_llava  # you can also specify the entry function (--action)
```

### Generating rules for geometric shapes / synthetic fossil samples

Running the following command can generate rules for geometric shapes in `dataset/rules.json`:

```shell
python run.py --module data.rule.generate --num_basic_geo_samples 10
```

or generate rules for synthetic fossil samples:

```shell
python run.py --module data.rule.generate --mode fossil --num_fossil_samples 10
```

Each data sample contains two parts:

- **shapes**: parameters and special information of each geometric shape.
- **relations**: relationship between two shapes in form of `[head_shape_idx, tail_shape_idx, relation_type]`

#### Example data sample

```json
{
  "shapes": [
    {
      "type": "line"
      //...
    },
    {
      "type": "ellipse"
      //...
    }
  ],
  "relations": [[0, 1, "tangent line"]]
}
```

You can control the generation process with the following arguments:

- max_num_shapes: the maximum number of shapes in each sample. Default is 10
  and there are arguments for controling the proportion of different shapes and relations, for example:
- polygon_shape_level: the proportion of polygon in all shapes
- line_shape_level: the proportion of line in all shapes
- ...
- polygon_tangent_line_level: the proportion of generating a tangent line in all polygon relations
- polygon_shared_edge_level: the proportion of generating a new polygon that have a shared edge with a given polygon
- ellipse_concentric_level: the proportion of generating a set of ellipses that is concentric with a given ellipse
- ...

Each 'level' argument is an integer (with a default value) representing the relative level within its shape/relation block. For more details, please refer to `RuleArgs` in `common/args.py`. All 'level' arguments will be transformed into probabilities using L1 normalization (sum normalization).

If more ellipse is expected, you can set a higher level for ellipse_shape_level:

```shell
python run.py --module data.rule.generate --polygon_shape_level 1 --line_shape_level 1 --ellipse_shape_level 3 --spiral_shape_level 1
```

### Running Module 'draw'

Two python files, `pil_backend.py` and `plt_backend.py` is provided, in which the former one is written in pillow, providing continuous change of shape, and a relatively less noisy image; the latter, in comparison, provides hand-drawing line style and more natural noise. `plt_backend.py` is recommended to use and `draw.py` will automatically use this version. You can change the preferred version by setting argument `backend` to `plt` or `pil`.

To use `plt_backend.py`, the following arguments are expected:

- rules: "list[dict[str, Any]]". Mandatory. The rules you would like to draw.
- random_seed: int|None. The default value is None. Control the random seed.
- randomize: bool. The default value is True. Enable the noise-applying procedure.
- size: "tuple[float, float]". The deault value is (6.4, 6.4).
- dpi: int. The default value is 100. dpi \* size = resolution.
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

To simply generate a picture with default settings, use the following command:

```shell
python run.py --module data.draw --backend plt
```

### Running caption

```shell
python run.py --module data.caption [ --caption_batchsize ${BatchSize} ] [ --caption_llm ${LLM ID} ] [ --numeric_ratio ${ratio} ]
```

Only part of the shapes will add numeric values, controlled by ${ratio}.

## Implementation detail

### Rule

generate random rules with conditions.

write how you generate rules here...

### Draw

draw shapes according to the rule.

write how you implement, e.g., adding noise...

### Caption

generate image captions according to the rule.

write how you convert rules to intermediate format and generate captions...

## Contributing

Fork and open a pull request. Follow the instructions below or your PR will fail.

1. Use `Pylance` (basic level) to lint your code while doing your work. Refer to https://docs.pydantic.dev/latest/integrations/visual_studio_code/#configure-vs-code to configure your VSCode. NOTE: Be cautious of using `# type: ignore` to suppress type errors, as you may be ignoring valuable traces of bugs; usually typing.cast() is more preferred.
2. Use `black` to format your code before opening a PR:

   ```shell
   pip install black
   black . --line-length 120 --extend-exclude llava
   ```

Note: If you want to add external modules which will not pass the linter, you can add them to `pyrightconfig.json` and `.github/workflows/lint_format.yaml`.

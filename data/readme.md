Data construction pipeline for fossil stage-1 (geobench) and stage-2.

## rule

generate random rules with conditions.

for rules generation for stage-1 (geobench):

```shell
python run.py --module data.rule.generate --stage 1 --num_basic_geo_samples 1000
```

more arguments about geometric shapes are provided in `RuleArgs` in `common/args.py`

for rules generation for stage-1 (geobench):

```shell
python run.py --module data.rule.generate --stage 2 --num_fossil_samples 100
```

## Draw

Draw shapes according to the rule.

In the default backend (which means `plt_backend.py`), the python script will receive the `rules.json` in path `dataset/rules.json` and handle each rule by turn. After handling all the rules, a basic image will be generated. Then, the script will add noise to the image and generate a final image. The noise here contains Gaussian noise and Perlin noise.

To run the script, use the following command:
```shell
python run.py --module data.draw.draw --backend plt --stage 1
```

More arguments are provided in `DrawArgs` in `common/args.py`. And you may also read readme file in the root directory for more details.

Currently, the Stable Diffusion part is not merged into the project. The script `diffusion_backend_new.py` is not available.

## caption

generate image captions according to the rule.

write how you convert rules to intermediate format and generate captions...

## vqa

generate VQA for geobench, according to the captions.

## format

format the data to llava trainable format.

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

## draw

draw shapes according to the rule.

write how you implement, e.g., adding noise...

## caption

generate image captions according to the rule.

write how you convert rules to intermediate format and generate captions...

## vqa

generate VQA for geobench, according to the captions.

## format

format the data to llava trainable format.

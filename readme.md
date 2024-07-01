# GeoCap

geometry caption & geography fossil caption

## install

Requirements are provided in `deploy/requirements.txt`. It's recommended to use python 3.10.

## running a module

A `run.py` is provided for running a module. This is an elegant workaround for importing errors from different packages. Examples:

```shell
python run.py --module data.rules --num_basic_geo_samples 10  # default entry is main()
python run.py --module data.format --action to_llava  # you can also specify the entry function (--action)
```

## Contributing

Fork and open a pull request. Follow the instructions below or your PR will fail.

1. Use `Pylance` (basic level) to lint your code while doing your work. Refer to https://docs.pydantic.dev/latest/integrations/visual_studio_code/#configure-vs-code to configure VSCode.
2. Use `black` to format your code before opening a PR:

   ```shell
   pip install black
   black . --line-length 120 --extend-exclude models
   ```

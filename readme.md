# GEOCAP

geometry caption & geography fossil caption

## Contributing

Fork and open a pull request. Follow the instructions below or your PR will fail.

1. Use `Pylance` (basic level) to lint your code. Refer to https://docs.pydantic.dev/latest/integrations/visual_studio_code/#configure-vs-code to configure with VSCode.
2. Use `black` to format your code.
   ```shell
   pip install black
   black . --line-length 120 --extend-exclude models
   ```

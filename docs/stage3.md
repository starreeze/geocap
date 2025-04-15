## Stage 3 Dataset Generation

Generate fossil caption data for stage 3 training.

```shell
./run -m stage3.generate_dataset \
    --save_data_path dataset/stage3_data/ \
    --num_replace_prompt_dir stage3/prompts/num_replace_icl.txt \
    --num_replace_llm api-deepseek-v3-0324 \
    --num_replace_batchsize 1 \
    --paraphrase_prompt_dir data/caption/prompts/paraphrase_stage3.txt \
    --caption_llm api-deepseek-v3-0324 \
    --caption_batchsize 1
```

The generation process contains the following steps:

1. Recognize visual features of the fossil images. The results (including original descriptions) are in `{save_data_path}/instructions_all.jsonl`.
2. Replace numerical information in original fossil description. The results are in `{save_data_path}/num_replace.jsonl`
3. Paraphrase the description to enhance diversity of description. The results are in `{save_data_path}/paraphrase.jsonl`
4. Tag format the description with `<feature>` and `</feature>` to categorize sentence into different features. The results are in `{save_data_path}/tag_format.jsonl`
5. Add default value to the description in case of missing features. The results are in `{save_data_path}/add_default_value.jsonl`

## Feature recognition for specific part of a fossil image.

### Initial Chamber

A class `ProloculusDetector` is designed to detect the initial chamber of a fossil image. It uses a sliding window with different window sizes to find the white circle in the central region of the image. Call the `detect_initial_chamber` method with the input image path to get the initial chamber-related features.

### Volutions

The `VolutionCounter` class in `volution_counter.py` is designed to detect and measure the volutions with a "adsorb-scan" strategy. Call the `count_volutions` method with the input image to get the volutions-related features.

### Chomatas

TODO

### Usage Example

The `recognize_feature` function in `recognize.py` is an example of feature recognition. It returns all features extracted by stage3 visual tools including initial chamber, volutions and chomatas.

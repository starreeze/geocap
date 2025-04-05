## Stage 3 Dataset Generation

Generate fossil caption data for stage 3 training.

```shell
./run -m feat_recognize.generate_dataset \
    --save_data_path dataset/ \
    --desc_prompt_dir feat_recognize/prompt_icl.txt \
    --desc_llm qwen25-14 \
    --desc_batchsize 1 \
    --paraphrase_prompt_dir data/caption/paraphrase_stage3.txt \
    --caption_llm api-deepseek-chat \
    --caption_batchsize 1
```

The generation process contains 3 parts:

1. Recognize visual features of the fossil images. The results are in `{save_data_path}/instructions.jsonl`.
2. Replace numerical information in original fossil description. The results are in `{save_data_path}/stage3.jsonl`
3. Paraphrase the description to enhance diversity of description. The results are in `{save_data_path}/stage3_paraphrase.jsonl`

`--desc_prompt_dir` is the prompt text file for numerical information replacement. `--desc_llm` specify the llm for replacement. Similarly, `--caption_prompt_dir` and `--caption_llm` specify the prompt text file and LLM for generating diverse paraphrased descriptions.

## Feature recognition for specific part of a fossil image.

### Initial Chamber

The `detect_initial_chamber` function in `initial_chamber.py` uses `cv2.houghcircle` to detect the circle in the central region of the image.

### Volutions

The `VolutionCounter` class in `volution_counter.py` is designed to detect and measure the volutions with a "adsorb-scan" strategy. Call the `count_volutions` method with the input image to get the volutions-related features, it also returns whether the initial chamber was detected with a high confidence level.

### Usage Example

The `recognize_feature` function in `recognize.py` is an example of feature recognition. It first recognizes the volutions in the fossil image, and try to detect the initial chamber twice with different confidence level.

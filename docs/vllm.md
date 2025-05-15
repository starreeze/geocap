## Evaluate different models

### API-based models

### Local models

Here are download commands for all supported models:

Models for general purpose:
1. BLIP2
    ```shell
    [TODO] huggingface-cli download ...
    ```
2. InstructBLIP
    ```shell
    [TODO] huggingface-cli download ...
    ```
3. Minigpt-v2
  - Download the official [MiniGPT](https://github.com/Vision-CAIR/MiniGPT-4) repository. Update the files `MiniGPT/minigpt4/configs/models/minigpt_v2.yaml` and `MiniGPT/eval_configs/minigptv2_eval.yaml` as instructed.
  - ```shell
    [TODO] huggingface-cli download ...
    ```
4. LLaVA-1.5
    ```shell
    [TODO] huggingface-cli download ...
    ```
5. LLaVA-OneVision
  - Download the official [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) repository and use the LLaVA library from there. Name the model as `llavaonevision-xx`.
  - ```shell
    [TODO] huggingface-cli download ...
    ```
6. mPLUG-Owl3
    ```shell
    huggingface-cli download mPLUG/mPLUG-Owl3-7B-240728 --local-dir models/mPLUG_Owl3-7B
    ```
7. InternVL-2.5
    ```shell
    huggingface-cli download OpenGVLab/InternVL2_5-8B --local-dir models/InternVL2_5-8B
    huggingface-cli download OpenGVLab/InternVL2_5-26B --local-dir models/InternVL2_5-26B
    huggingface-cli download OpenGVLab/InternVL2_5-78B --local-dir models/InternVL2_5-78B
    ```
8. MiniCPM-V-2.6
    ```shell
    huggingface-cli download openbmb/MiniCPM-V-2_6 --local-dir models/MiniCPM_V_2_6-8B --token <YOUR_HF_TOKEN>
    ```
9. GLM-4V
    ```shell
    huggingface-cli download THUDM/glm-4v-9b --local-dir models/GLM_4V-9B
    ```
10. Mantis-Idefics2    
    ```shell
    huggingface-cli download TIGER-Lab/Mantis-8B-Idefics2 --local-dir models/Mantis-Idefics2-8B
    ```       
11. Qwen2-VL
    ```shell
    huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir models/Qwen2_VL_Instruct-7B
    huggingface-cli download Qwen/Qwen2-VL-72B-Instruct --local-dir models/Qwen2_VL_Instruct-72B
    ```       
12. LLaMA-3.2-Vision
    ```shell
    huggingface-cli download meta-llama/Llama-3.2-90B-Vision-Instruct --local-dir models/Llama_3_2_Vision_Instruct-90B --token <YOUR_HF_TOKEN>
    ```     

Models for special purpose:
13. G-LLaVA
    ```shell
    huggingface-cli download renjiepi/G-LLaVA-7B --local-dir models/G_LLaVA-7B
    huggingface-cli download renjiepi/G-LLaVA-13B --local-dir models/G_LLaVA-13B
    ```     
14. Math-LLaVA
    ```shell
    huggingface-cli download Zhiqiang007/Math-LLaVA --local-dir models/Math_LLaVA-13B
    ```     
15. Math-PUMA
    ```shell
    huggingface-cli download Math-PUMA/Math-PUMA_Qwen2VL-7B --local-dir models/Math_PUMA_Qwen2_VL-7B
    ```     
16. QVQ
    ```shell
    huggingface-cli download Qwen/QVQ-72B-Preview --local-dir models/QVQ-72B
    ```     

The default download path is `models`. You can modify this path, but make sure to update it consistently in all subsequent locations.  

For certain models, the owners have set special licenses that require users to submit an application and obtain approval before downloading. Please first log in the [HuggingFace](https://huggingface.co/), locate the corresponding model repository, and submit an application. Once approved, generate an access token with download permission and replace `<YOUR_HF_TOKEN>` in the download command above to proceed with the download.

You can use the following command to evalutate anyone of above models:
```shell
./run -m eval.gepbench \
    --eval_model <MODEL_NAME> \
    --eval_batchsize <EVAL_BATCHSIZE> \
    --vqa_question_dir <QUESTION_DIR> \
    --vqa_output_dir <OUTPUT_DIR> \
    --figure_dir <FIGURE_DIR> \
    --figure_prefix <FIGURE_PREFIX> \
    --start_pos <START_POS> \
    --end_pos <END_POS>
```

`<MODEL_NAME>` can be one of `llava-1.5-13b`, `llava-1.5-7b-20k`, `llava-v1.6-34b-hf`, `llava-1.5-13b-1`, `llava-1.5-7b-geo`, `llava-v1.6-mistral-7b-hf`, `llava-1.5-13b-geo`, `llava-1.5-7b-hf`, `llava-v1.6-vicuna-13b-hf`, `llava-1.5-13b-hf`, `llava-1.5-7b-merge`, `llava-v1.6-vicuna-7b-hf`, `llava-1.5-13b-merge`, `llava-next-110b-hf`, `llava-1.5-7b`, `llava-next-72b-hf`, `minigptv2`, `blip2-flan-t5-xl`, `blip2-opt-2.7b`, `blip2-opt-6.7b-coco`, `blip2-flan-t5-xxl`, `blip2-opt-6.7b`, `instructblip-flan-t5-xl`, `instructblip-vicuna-7b`, `instructblip-flan-t5-xxl`, `internlm2-chat-7b`, `instructblip-vicuna-13b`, `G_LLaVA-13B`, `G_LLaVA-7B`, `GLM_4V-9B`, `InternVL2_5-26B`, `InternVL2_5-78B`, `InternVL2_5-8B`, `Llama_3_2_Vision_Instruct-90B`, `Mantis_Idefics2-8B`, `Math_LLaVA-13B`, `Math_PUMA_Qwen2_VL-7B`, `MiniCPM_V_2_6-8B`, `mPLUG_Owl3-7B`, `QVQ-72B`, `Qwen2_VL_Instruct-72B`, `Qwen2_VL_Instruct-7B`.

`<EVAL_BATCHSIZE>` should be a positive integer with default `4`.

We generated two difficulty levels of questions, `easy` and `hard`, which are stored by default in `dataset/vqa-easy` and `dataset/vqa-hard`, respectively. 
So, `<QUESTION_DIR>` should refer to one of these two paths and `<FIGURE_DIR>` should be `dataset/figures`.

`<FIGURE_PREFIX>` denotes the filename prefix of the images corresponding to the evaluation questions. These prefixes indicate which difficulty level the images belong to, usually being either `easy` or `hard`. Additionally, `en` represents the images from the easy-difficulty questions after noise has been added.

`<OUTPUT_DIR>` refers to the folder where evaluation results are saved. Your evaluation results on a particular model will be recorded in a subfolder under `<OUTPUT_DIR>` that shares the model's name. This subfolder will contain the model's evaluation results (recorded the whole predictions) across all aspects of GePBench, along with a summary CSV file providing accuracy rates. You can set `<OUTPUT_DIR>` as one of `results/easy` and `results/hard`. But you can change the path if you don't want to overwrite the results in last run.

`<START_POS>` and `<END_POS>` determine the range of image IDs for which the question will be used to test the model. The defaults are `0` and `20000`.

Here's an example,
```shell
./run -m eval.gepbench \
    --eval_model mPLUG_Owl3-7B \
    --eval_batchsize 16 \
    --vqa_question_dir dataset/vqa-easy \
    --vqa_output_dir results/easy \
    --figure_dir dataset/figures \
    --figure_prefix easy \
    --start_pos 0 \
    --end_pos 1000
```


## Analyzing the Impact of Visual Encoders

These are the evaluation codes for the "Impact of Visual Encoders" section in the paper. The `llavav.py` file evaluates a single visual encoder, while `llavam.py` evaluates a hybrid visual encoder.

At first, download the official [Law_of_Vision_Representation_in_MLLMs](https://github.com/bronyayang/Law_of_Vision_Representation_in_MLLMs) repository, and use the LLaVA library there. Follow the instructions to download [model](https://huggingface.co/shijiay). Name the single visual encoder models as `llavav-xx` and the hybrid visual encoder models as `llavam-xx`.

Run the following code for evaluation:
```shell
./run -m eval.gepbench \
    --eval_model <MODEL_NAME> \
    --eval_batchsize <EVAL_BATCHSIZE> \
    --vqa_question_dir <QUESTION_DIR> \
    --vqa_output_dir <OUTPUT_DIR> \
    --figure_dir <FIGURE_DIR> \
    --figure_prefix <FIGURE_PREFIX> \
    --start_pos <START_POS> \
    --end_pos <END_POS>
```

`<MODEL_NAME>` can be one of `llavav-1.5-7b-clip`, `llavav-1.5-7b-siglip`, `llavav-llama3-Instruct`, `llavav-1.5-7b-clip224`, `llavav-geo-7b`, `llavav-openhermes`, `llavav-1.5-7b-dinov2`, `llavav-llama3`, `llavav-1.5-7b-openclip`, `llavav-llama3.1`, `llavam-1.5-7b-clipdino224`, `llavam-1.5-7b-sd1.5`, `llavam-1.5-7b-sdim`, `llavam-1.5-7b-clipdino336`, `llavam-1.5-7b-sd2.1`, `llavam-1.5-7b-sdxl`, `llavam-1.5-7b-dit`, `llavam-1.5-7b-sd3`.

Other fields comply with requirements consistent with the general evaluation process mentioned above.


For example, 
```
./run -m eval.gepbench --eval_model llavam-xx --eval_batchsize 8 --vqa_question_dir dataset/dataset/vqa-hard --vqa_output_dir results/hard --figure_dir dataset/dataset/figures --figure_prefix hard --end_pos 2000
```


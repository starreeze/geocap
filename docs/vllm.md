## Minigpt-v2

1. Download the official [MiniGPT](https://github.com/Vision-CAIR/MiniGPT-4) repository and update the files `MiniGPT/minigpt4/configs/models/minigpt_v2.yaml` and `MiniGPT/eval_configs/minigptv2_eval.yaml` as instructed.
2. Run the code for evaluation:

```
./run -m eval.evaluate --eval_model minigptv2 --eval_batchsize 8 --vqa_question_dir dataset/dataset/vqa-hard --vqa_output_dir results/hard --figure_dir dataset/dataset/figures --figure_prefix hard --end_pos 2000
```

## LLaVA

In our work, we evaluate different variants of LLaVA, such as llava-1.5, llava-onevision, and llava with different visual encoders. Therefore, during the evaluation process, we need to import different LLaVA libraries, for example, `from llava.model.builder import load_pretrained_model`. The corresponding LLaVA library depends on the type of LLaVA model.

### llava.py

This is the evaluation code for the LLaVA-v1.5 series models. The LLaVA library used in the code can be directly taken from the `llava` directory in our project.

### llavam.py and llavav.py

These are the evaluation codes for the "Impact of Visual Encoders" section in the paper. The `llavav.py` file evaluates a single visual encoder, while `llavam.py` evaluates a hybrid visual encoder.

1. Download the official [Law_of_Vision_Representation_in_MLLMs](https://github.com/bronyayang/Law_of_Vision_Representation_in_MLLMs) repository, and use the LLaVA library from there. Follow the instructions to download the [model](https://huggingface.co/shijiay). Name the single visual encoder models as `llavav-xx` and the hybrid visual encoder models as `llavam-xx`.
2. Run the code for evaluation:

```
./run -m eval.evaluate --eval_model llavav-xx/llavam-xx --eval_batchsize 8 --vqa_question_dir dataset/dataset/vqa-hard --vqa_output_dir results/hard --figure_dir dataset/dataset/figures --figure_prefix hard --end_pos 2000
```

### llavaonevision.py

This is the evaluation code for the LLaVA-OneVision series models.

1. Download the official [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) repository and use the LLaVA library from there. Name the model as `llavaonevision-xx`.
2. Run the code for evaluation:

```
./run -m eval.evaluate --eval_model llavaonevision-xx --eval_batchsize 8 --vqa_question_dir dataset/dataset/vqa-hard --vqa_output_dir results/hard --figure_dir dataset/dataset/figures --figure_prefix hard --end_pos 2000
```

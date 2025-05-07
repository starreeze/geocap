# Train

## Pretrain

For the pretraining stage:
1.Run `easy_data.py`, `hard_data.py`, and `merge_data`.py in `LLaVA-1.5-GeP/train/pretrain` to prepare the data.
2.Execute the script `LLaVA-1.5-GeP/train/pretrain/pretrain.sh`

## Finetune

For the finetuning stage:
1.Run `easy_data.py`, `hard_data.py`, and `merge_data.py` in `LLaVA-1.5-GeP/train/finetune` to prepare the data.
2.Execute the script `LLaVA-1.5-GeP/train/pretrain/finetune.sh`

# Evaluation

In GePBench, we evaluate LLaVA-1.5-7b and LLaVA-1.5-GeP-7b on 11 benchmarks from LLaVA itself, as well as 6 benchmarks in the Math and Chart domains. We primarily use the official toolkit or server for the evaluation.

## Scripts for LLaVA offical benchmark

For LLaVA offical benchmark,you can refer to the official evaluation instructions in [Evaluation.md](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md?plain=1).

### LLaVA-Bench-in-the-Wild

As stated in our paper, we use Qwen-2.5-14B-Instruct instead of GPT-4 as the evaluation model. Therefore, please copy `./LLaVA-1.5-GeP/eval/scripts/eval_qwen_review_bench.py` and `./LLaVA-1.5-GeP/eval/scripts/summarize_qwen_review.py` to `./llava/eval`

## Scripts for Document and Diagrams Interpretation Benchmark

### DocVQA
1. Prepare Dataset: Download from [Huggingface](https://huggingface.co/datasets/lmms-lab/DocVQA) and save into ./DocVQA
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3  bash ./LLaVA-1.5-GeP/eval/scripts/DocVQA.sh
```

### CharXiv
1. Download the [CharXiv](https://github.com/princeton-nlp/CharXiv) repository and follow the instructions on GitHub to prepare the evaluation data.
2. Copy the `descriptive_utils.py`, `evaluate.py`, `generate.py` and `reasoning_utils.py` files from the `./LLaVA-1.5-GeP/eval/CharXiv` path to the CharXiv official repository's `CharXiv/src` folder
3. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3  bash ./LLaVA-1.5-GeP/eval/scripts/CharXiv.sh
```

### AI2D
We evaluated AI2D using VLMEvalKit.
1. Visit the VLMEvalKit [Quickstart Guide](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) for installation instructions. or you can run the following command for a quick start:
```Shell
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3  python run.py --data AI2D_TEST --model llava_v1.5_7b --verbose
```

### ChartBench
1. Download the [ChartBench](https://github.com/IDEA-FinAI/ChartBench) repository and follow the instructions on GitHub to prepare the evaluation data.
2. Copy the `qwen_filter.py`, and `utils.py` files from the `./LLaVA-1.5-GeP/eval/ChartBench` path to the ChartBench official repository's `ChartBench/Stat` folder
3. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./LLaVA-1.5-GeP/eval/scripts/ChartBench.sh
```

### ChartQA
1. Download the [ChartQA](https://github.com/vis-nlp/ChartQA) repository and follow the instructions on GitHub to prepare the evaluation data.
2. Copy the `extract_answer.py`, `get_response.py`, and `score.py` files from the `./LLaVA-1.5-GeP/eval/ChartQA` path to the ChartQA official repository's `ChartQA/ChartQADataset/test` folder
3. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./LLaVA-1.5-GeP/eval/scripts/ChartQA.sh
```

## Scripts for Mathematical Reasoning benchmark

### MATH-V

1. Download the [MATH-V](https://github.com/mathllm/MATH-V) repository and follow the instructions on GitHub to prepare the evaluation data.
2. Replace the files `evaluate.py`, `get_response.py`, and `utils.py` in the MATH-V official repository's `evaluation` folder with the ones from the `./LLaVA-1.5-GeP/eval/MATH-V` path
3. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./LLaVA-1.5-GeP/eval/scripts/MATH-V.sh
```

### MathVerse
1. Download the [MathVerse](https://github.com/ZrrSkywalker/MathVerse) repository and follow the instructions on GitHub to prepare the evaluation data.
2. Copy the `extract_answer.py`, `get_response.py`, and `score.py` files from the `./LLaVA-1.5-GeP/eval/MathVerse` path to the MathVerse official repository's `evaluation` folder
3. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./LLaVA-1.5-GeP/eval/scripts/MathVerse.sh
```

### MathVista
1. Download the [MathVista](https://github.com/lupantech/MathVista) repository and follow the instructions on GitHub to prepare the evaluation data.
2. Copy the `extract_answer.py`, `get_response.py`, and `score.py` files from the `./LLaVA-1.5-GeP/eval/MathVista` path to the MathVista official repository
3. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./LLaVA-1.5-GeP/eval/scripts/MathVista.sh
```

### We-Math
We evaluated We-Math using VLMEvalKit.
1. Visit the VLMEvalKit [Quickstart Guide](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) for installation instructions. or you can run the following command for a quick start:
```Shell
git clone https://github.com/open-compass/VLMEvalKit.git
cd VLMEvalKit
pip install -e .
```
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3  python run.py --data WeMath --model llava_v1.5_7b --verbose
```

## Scripts for Medical Image Analysis benchmark

### PMC-VQA
1. Prepare Dataset: Download from [Huggingface](https://huggingface.co/datasets/RadGenome/PMC-VQA) and save into ./PMC-VQA
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 
bash ./LLaVA-1.5-GeP/eval/scripts/PMC-VQA.sh
```

### SLAKE
1. Prepare Dataset: Download from [Huggingface](https://huggingface.co/datasets/BoKelvin/SLAKE) and save into ./SLAKE
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 
bash ./LLaVA-1.5-GeP/eval/scripts/SLAKE.sh
```

### PathVQA
1. Prepare Dataset: Download from [Huggingface](https://huggingface.co/datasets/flaviagiammarino/path-vqa) and save into ./PathVQA
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 
bash ./LLaVA-1.5-GeP/eval/scripts/PathVQA.sh
```

### MedXpertQA
1. Prepare Dataset: Download from [Huggingface](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) and save into ./MedXpertQA
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 
bash ./LLaVA-1.5-GeP/eval/scripts/MedXpertQA.sh
```
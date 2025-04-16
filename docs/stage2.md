# 二阶段画图部署

## MimicBrush

* 下载MimicBrush并配置其使用的相关环境

[https://github.com/ali-vilab/MimicBrush](https://github.com/ali-vilab/MimicBrush)

## 部署运行

* 将`data/draw/diffusion/`下的所有文件放入MimicBrush项目根目录下（注意run_gradio3_demo.py要覆盖原有的版本）
* 按需修改`cpu_test.sh`和`gpu_test.sh`中的相关路径
* `cd MimicBrush/`
* `sh cpu_test.sh <rules-{keyword}.json中的keyword>`，生成best_match.txt，保存位置在`rules所在目录/{keyword}/best_match.txt`
* `sh gpu_test.sh <keyword>`，生成图片，注意按需修改`CUDA_VISIBLE_DEVICES`，保存位置在`rules所在目录/{keyword}/pics`
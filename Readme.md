# AI4SC 项目镜像快速使用
镜像源：https://hub.docker.com/r/zzyharry/ai4sc
拉取镜像：
```
docker pull zzyharry/ai4sc:cu118
```

目标：提供最短路径，一键运行训练与评估。将 `<INPUT_DIR>`、`<OUTPUT_DIR>` 替换为你的宿主机目录；有 GPU 保留 `--gpus all`，无 GPU 删掉即可。

## 一、训练（Quick Start）
- 路径约定（容器内固定）：
  - 输入数据：`/app/data`
  - 训练输出：`/app/train_output`
- 一键训练（示例路径，按需替换）：
```
docker run --rm --gpus all -it\
  --user $(id -u):$(id -g) \
  -e HOME=/app/train_output \
  -e XDG_CACHE_HOME=/app/train_output/.cache \
  -e MPLCONFIGDIR=/app/train_output/.mpl \
  -v <INPUT_DIR>:/app/data \
  -v <OUTPUT_DIR>:/app/train_output \
  zzyharry/ai4sc:cu118 bash -lc 'mkdir -p /app/train_output/.mpl /app/train_output/.cache && /opt/conda/envs/ai4sc/bin/python /app/scripts/train.py --data-dir /app/data --epochs 500 --plot-every 1'
```
- 提示：
  - 使用 `--user $(id -u):$(id -g)` 确保输出目录可写。
  - 首次 CUDA/cuDNN 初始化可能较慢；无 GPU 环境直接去掉 `--gpus all`。
  - 训练中可以在<OUTPUT_DIR>查看image文件夹，其中会保存训练过程中的loss曲线图片，和val的可视化图。如果不希望每一个epoch都绘制的话，可以将`--plot-every`设置为一个较大的值。
  - 如果不希望训练太多epoch的话，可以将`--epochs`设置为一个较小的值。

## 二、评估（test.py）
- 前置：训练主目录（如 `train_results_YYYYMMDD_HHMMSS`）中存在 `Par.json` 与 `models/best_model.pdparams`。
- 评估输出建议挂载到 `/app/test_output`，避免默认目录权限错误。
- 一键评估（按需替换路径与训练目录）：
- 将下面的<INPUT_DIR>，<OUTPUT_DIR>（两处），<train_results_YYYYMMDD_HHMMSS>（这是train生成的）替换为你自己的路径。
```
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -e HOME=/app/train_output -e XDG_CACHE_HOME=/app/train_output/.cache -e MPLCONFIGDIR=/app/train_output/.mpl \
  -v <INPUT_DIR>:/app/data \
  -v <OUTPUT_DIR>:/app/train_output \
  -v <OUTPUT_DIR>:/app/test_output \
  zzyharry/ai4sc:latest bash -lc 'mkdir -p /app/train_output/.mpl /app/train_output/.cache && /opt/conda/envs/ai4sc/bin/python /app/scripts/test.py --ux /app/data/UX_nan_filtered.npy --uy /app/data/UY_nan_filtered.npy --mask /app/data/mask.npy --train_dir /app/train_output/<train_results_YYYYMMDD_HHMMSS> --output /app/train_output/test_report_$(date +%Y%m%d_%H%M%S) --batch_size 4'
```

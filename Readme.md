# 1、AI4SC 项目镜像快速使用
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

### 评估脚本特点
- **多维度评估**：包含误差指标、物理量指标、频谱分析、技能评分等
- **丰富的可视化**：生成20+种图表，涵盖误差曲线、物理量对比、涡度分析、能谱分析等
- **详细的空间分析**：提供3个典型案例的速度场逐帧对比和残差分析
- **性能评估**：包含推理速度和吞吐量测试
- **标准化输出**：所有结果按时间戳组织，便于对比和追踪
# 2、训练输出文件说明
## 一、训练输出目录结构
训练脚本会在 `train_output/` 目录下生成完整的训练结果，目录结构如下：

```
train_output/
└── train_results_YYYYMMDD_HHMMSS/     # 单次训练的主目录（时间戳命名）
    ├── Par.json                       # 训练参数配置文件
    ├── models/
    │   └── best_model.pdparams        # 验证集最优模型权重
    ├── images/                        # 训练过程可视化图片
    │   ├── train_val_loss_curve.png  # 损失曲线图
    │   ├── pred_vs_gt_epoch_XXX.png  # 预测vs真实对比图（按epoch）
    │   └── ...
    ├── loss_curve.csv                 # 训练和验证损失数据
    ├── seed_23/                       # 随机种子目录
    │   └── *.log                      # 训练日志文件
    └── .cache/                        # Paddle框架缓存（自动创建）
        ├── paddle/
        └── ...
```

### 文件详细说明

#### 1. 核心配置文件
- **Par.json**: 包含所有训练参数，包括模型架构、数据维度、损失函数权重等
  - 用途：模型复现、评估脚本调用、超参数记录
  - 生成时机：训练开始时一次性写入
  - 关键参数：`nx`, `ny`, `nf`, `lb`, `lf`, `mse_weight`, `div_weight`, `momentum_weight`

#### 2. 模型文件
- **models/best_model.pdparams**: 训练过程中验证集损失最小的模型权重
  - 保存策略：每当验证集损失创新低时自动保存
  - 用途：模型评估、推理、继续训练
  - 文件格式：PaddlePaddle二进制格式

#### 3. 可视化文件
- **images/**: 训练过程可视化图片目录
  - `train_val_loss_curve.png`: 训练和验证损失曲线
  - `pred_vs_gt_epoch_XXX.png`: 每个epoch的预测结果对比图
  - 生成频率：由 `--plot-every` 参数控制（默认每20个epoch）

#### 4. 数据文件
- **loss_curve.csv**: 训练和验证损失的历史记录
  - 格式：`epoch,train_loss,val_loss`
  - 用途：后续分析、绘制更详细的损失曲线

#### 5. 缓存目录
- **.cache/**: Paddle框架运行时缓存
  - `.mpl/`: Matplotlib配置缓存
  - `paddle/`: Paddle模型缓存和临时文件
  - 用途：加速重复运行，避免权限问题

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 500 | 训练轮数 |
| `--plot-every` | 20 | 绘图频率（epoch）|
| `--grad-log-every` | 0 | 梯度日志频率（batch）|
| `--grad-log-epochs` | 5 | 记录梯度的epoch数 |
| `--num-workers` | 0 | 数据加载进程数 |

# 3、评估输出文件说明
## 一、评估输出目录结构
评估脚本会在 `test_output/` 目录下生成完整的评估结果，目录结构如下：

```
test_output/
└── test_results_YYYYMMDD_HHMMSS/     # 单次评估的主目录（时间戳命名）
    ├── summary_test.json              # 评估结果摘要
    ├── metrics_frame.csv              # 逐帧误差指标
    ├── metrics_physics.csv            # 物理量评估指标
    ├── pred_test.npy                  # 预测结果数据
    ├── images_test/                   # 评估可视化图片
    │   ├── curves_metrics.png        # 误差曲线图
    │   ├── curves_physics.png        # 物理量曲线图
    │   ├── error_heatmap.png         # 误差热力图
    │   ├── vorticity_tau_X.png       # 涡度可视化（多时间点）
    │   ├── spectrum_tau_X.png        # 能谱对比图（多时间点）
    │   └── case_velocity_grids/       # 速度场详细分析
    │       ├── case_0/               # 第1个案例的详细图
    │       │   ├── tau_1_Ux_true.png
    │       │   ├── tau_1_Ux_pred.png
    │       │   ├── tau_1_Ux_residual.png
    │       │   ├── tau_1_Uy_true.png
    │       │   ├── tau_1_Uy_pred.png
    │       │   └── tau_1_Uy_residual.png
    │       ├── case_1/               # 第2个案例
    │       ├── case_2/               # 第3个案例
    │       ├── colorbar_reference_vertical.png    # 速度色标参考
    │       └── colorbar_residual_vertical.png     # 残差色标参考
    └── spectrum_tau_X.npz             # 能谱数据文件（多时间点）
```

### 文件详细说明

#### 1. 核心结果文件
- **summary_test.json**: 评估结果总摘要
  - 包含：测试损失、运行时间、首尾帧RMSE、Skill评分、推理速度指标
  - 用途：快速了解模型整体性能
  - 示例内容：`{"test_loss": 0.0123, "RMSE_mean_first": 0.045, "Skill_mean_last": 0.89}`

- **metrics_frame.csv**: 逐帧误差统计（18列指标）
  - 包含：RMSE、相对RMSE、MAE、L∞误差、SSIM、Skill等随时间变化
  - 格式：`tau,RMSE_mean,RMSE_median,RMSE_std,relRMSE_mean,...,Skill_std`
  - 用途：分析误差随预测时间的变化趋势

- **metrics_physics.csv**: 物理量评估（15列指标）
  - 包含：涡度RMSE、散度均值、散度RMSE、能量、涡度平方等
  - 格式：`tau,Vorticity_RMSE_mean,Vorticity_RMSE_median,...,Enstrophy_pred_mean`
  - 用途：评估模型在物理守恒性方面的表现

#### 2. 可视化分析
- **curves_metrics.png**: 误差曲线图
  - 展示：RMSE、MAE、Skill随预测时间的变化
  - 用途：直观了解模型预测精度衰减

- **curves_physics.png**: 物理量曲线图
  - 展示：涡度RMSE、真实与预测涡度平方对比
  - 用途：分析物理量预测准确性

- **error_heatmap.png**: 误差热力图
  - 展示：不同样本在不同时间的RMSE分布
  - 用途：识别预测困难的样本和时间段

- **vorticity_tau_X.png**: 涡度可视化对比（tau=1,2,5,10）
  - 展示：真实涡度、预测涡度、涡度误差的三联图
  - 用途：空间分布误差分析

#### 3. 速度场详细分析
- **case_velocity_grids/**: 3个典型案例的详细分析
  - 每个案例包含40张图（前10个时间点×4种类型）
  - 类型：Ux_true, Ux_pred, Ux_residual, Uy_true, Uy_pred, Uy_residual
  - 残差图使用seismic色标，便于观察正负误差
  - 提供全局色标参考图，确保不同案例间可比性

#### 4. 频谱分析
- **spectrum_tau_X.npz**: 能谱数据（tau=1,2,5,10）
  - 包含：波数k、真实能谱E_true、预测能谱E_pred、相对误差dE
  - 用途：分析模型在不同尺度上的预测能力

- **spectrum_tau_X.png**: 能谱对比图
  - 展示：真实与预测能谱的log-log对比
  - 用途：评估模型对湍流能谱的复现能力

#### 5. 预测数据
- **pred_test.npy**: 最终预测结果
  - 格式：numpy数组，形状为[N, T, C, nx, ny]
  - 用途：后续分析、可视化、与其他方法对比

### 评估指标说明

| 指标类型 | 具体指标 | 物理意义 |
|----------|----------|----------|
| **误差指标** | RMSE | 均方根误差，衡量整体预测精度 |
| | 相对RMSE | 归一化RMSE，消除量纲影响 |
| | MAE | 平均绝对误差，对异常值更鲁棒 |
| | L∞ | 最大绝对误差，反映最坏情况 |
| | SSIM | 结构相似性，考虑空间结构 |
| | Skill | 技能评分，相对于基线的改进 |
| **物理指标** | 涡度RMSE | 旋转特性的预测精度 |
| | 散度均值 | 质量守恒性评估 |
| | 散度RMSE | 散度场预测精度 |
| | 能量密度 | 动能守恒性评估 |
| | 涡度平方 | 涡度强度预测 |
| **性能指标** | 推理速度 | 单次前向传播耗时 |
| | 吞吐量 | 每秒处理样本数 |

### 评估参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch_size` | 10 | 评估批大小，影响内存使用和速度 |
| `--LF` | 20 | 评估预测窗口长度，控制预测时间步数 |
| `--train_dir` | - | 训练输出目录，自动查找模型和配置 |
| `--output` | - | 评估输出目录，默认生成时间戳子目录 |
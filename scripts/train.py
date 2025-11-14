"""
训练脚本（参数化数据目录与轮数）。

- 在导入第三方库前设置缓存与配置目录，确保跨平台与容器环境下的可写性。
- 保持原训练管线、模型、数据切分、损失与调度逻辑不变。
- 仅规范代码结构并补充说明性模块文档，便于后续复现与维护。
"""

import json
import logging
import math
import os
import random
import sys
import time
import argparse
import pathlib

# 确保缓存与配置目录在导入第三方库前即为可写路径，避免 ~/.cache 权限问题
try:
    _project_root_early = pathlib.Path(__file__).resolve().parents[1]
    # 强制将 HOME 指向项目内可写输出目录（容器中为 /app/train_output）
    _default_cache_root = str(_project_root_early / "train_output")
    os.makedirs(_default_cache_root, exist_ok=True)
    os.environ["HOME"] = _default_cache_root
    os.environ["XDG_CACHE_HOME"] = os.path.join(_default_cache_root, ".cache")
    os.environ["PADDLE_HOME"] = os.path.join(os.environ["XDG_CACHE_HOME"], "paddle")
    os.environ["MPLCONFIGDIR"] = os.path.join(_default_cache_root, ".mpl")
    for _d in [os.environ["XDG_CACHE_HOME"], os.environ["PADDLE_HOME"], os.environ["MPLCONFIGDIR"]]:
        try:
            os.makedirs(_d, exist_ok=True)
        except Exception:
            pass
except Exception:
    # 如果早期设置失败，至少保证后续不崩溃
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paddle
from tqdm import tqdm
import multiprocessing as mp

# 避免在 fork 子进程中初始化 CUDA 导致报错，强制使用 spawn
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


import sys, pathlib
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
# 基准模型：与原训练脚本一致
from src.model import Unet2D_with_FNO_without_atte_FCRB
# 确保可以从项目根导入 src 包（当从 scripts/ 目录运行时）
import pathlib as _pl
_script_dir = _pl.Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
# 统一导入共享训练工具（与基准保持一致）
from src.utils import (
    setup_seed,
    init_all,
    make_plot,
    CustomLoss,
    YourDataset_train,
    YourDataset,
    preprocess_train,
    preprocess,
    combined_scheduler,
    rollout,
)


if __name__ == "__main__":
    # CLI 参数：数据目录与训练轮数
    script_dir = pathlib.Path(__file__).resolve().parent
    project_root = script_dir.parent
    parser = argparse.ArgumentParser(description="Training script with configurable dataset directory and epochs")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(project_root / "data"),
        help="Directory containing preprocessed files: UX_nan_filtered.npy, UY_nan_filtered.npy, mask.npy",
    )
    parser.add_argument("--ux", type=str, default=None, help="Override path to UX_nan_filtered.npy")
    parser.add_argument("--uy", type=str, default=None, help="Override path to UY_nan_filtered.npy")
    parser.add_argument("--mask", type=str, default=None, help="Override path to mask.npy")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs (default: 500)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes (default: 0)",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=20,
        help="Plot frequency in epochs; set 0 to disable interim plots",
    )
    parser.add_argument(
        "--grad-log-every",
        type=int,
        default=0,
        help="Gradient norm log frequency in batches; 0 to disable",
    )
    parser.add_argument(
        "--grad-log-epochs",
        type=int,
        default=5,
        help="Log gradient norms only in the first K epochs",
    )
    args = parser.parse_args()

    # 创建日志与输出目录（与基准保持一致）
    seed_value = 23
    data_type = "float32"

    # 默认将训练结果写入项目根目录下的 train_output
    output_root_dir = project_root / "train_output"
    output_root_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    main_output_dir = str(output_root_dir / f"train_results_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    save_dir = os.path.join(main_output_dir, f"seed_{seed_value}")
    logger = init_all(seed_value, name=save_dir, dtype=data_type)

    # 数据路径（参数化，默认使用项目 data 目录）
    data_dir = pathlib.Path(args.data_dir).resolve()
    ux_data_dir = args.ux or str(data_dir / "UX_nan_filtered.npy")
    uy_data_dir = args.uy or str(data_dir / "UY_nan_filtered.npy")
    mask_dir = args.mask or str(data_dir / "mask.npy")
    logger.info(f"Data paths: ux={ux_data_dir}, uy={uy_data_dir}, mask={mask_dir}")

    # 加载与预筛选
    begin_time = time.time()
    traj_ux = np.load(ux_data_dir)
    traj_uy = np.load(uy_data_dir)
    traj_ux = np.expand_dims(traj_ux, axis=0)
    traj_uy = np.expand_dims(traj_uy, axis=0)
    if traj_ux.shape != traj_uy.shape:
        logger.info(f"! Ux and Uy shapes differ: {traj_ux.shape} vs {traj_uy.shape}")
        sys.exit()
    traj = np.stack([traj_ux, traj_uy], axis=2)  # [N, T, C, nx, ny]

    bad_timesteps = []
    for t in range(traj.shape[1]):
        frame = traj[0, t]
        if np.abs(frame).max() > 10.0:
            logger.info(f"! Time step {t} has extreme values: max={np.abs(frame).max():.2e}")
            bad_timesteps.append(t)
    if bad_timesteps:
        logger.info(f"Bad time steps: {bad_timesteps}")
        sys.exit()

    logger.info(f"Mask path: {mask_dir}")

    mask = np.load(mask_dir).reshape(1, 1, traj.shape[-2], traj.shape[-1])
    traj = traj * mask
    logger.info(f"Data loading time: {time.time() - begin_time:.2f}s")

    # 切分数据（与基准一致）
    nt_all = traj.shape[1]
    train_end = int(nt_all * 0.8)
    val_end = int(nt_all * 0.9)
    train_end = max(1, min(train_end, nt_all - 2))
    val_end = max(train_end + 1, min(val_end, nt_all - 1))
    traj_train = traj[:, :train_end]
    traj_val = traj[:, train_end:val_end]
    traj_test = traj[:, val_end:]

    logger.info(f"Shape of whole data (traj): {traj.shape}")
    logger.info(f"Shape of train data (traj_train): {traj_train.shape}")
    logger.info(f"Shape of val data (traj_val): {traj_val.shape}")
    logger.info(f"Shape of test data (traj_test): {traj_test.shape}\n")

    ################## 初始化核心参数字典 Par（与基准一致，仅消融物理损失） ##############################
    Par = {}
    Par["nx"] = traj_train.shape[-2]
    Par["ny"] = traj_train.shape[-1]
    Par["nf"] = 2
    Par["d_emb"] = 128

    logger.info(f"Dimension of flow (nx*ny): ({Par['nx']}, {Par['ny']})")
    logger.info(f"Number of features (nf): {Par['nf']}")

    Par["lb"] = 10
    Par["lf"] = 2
    Par["LF"] = 10
    Par["channels"] = Par["nf"] * Par["lb"]
    Par["num_epochs"] = int(args.epochs)
    logger.info(f"Number of timesteps as inputs (lb): {Par['lb']}")
    logger.info(f"Number of timesteps as outputs (lf): {Par['lf']}")
    logger.info(f"Number of timesteps for long-term prediction (LF): {Par['LF']}")
    logger.info(f"Number epochs: {Par['num_epochs']}\n")

    time_cond = np.linspace(0, 1, Par["lf"])
    if Par["lf"] == 1:
        time_cond = np.linspace(0, 1, Par["lf"]) + 1

    t_min = np.min(time_cond)
    t_max = np.max(time_cond)
    if Par["lf"] == 1:
        t_min = 0
        t_max = 1

    Par["inp_shift"] = float(np.mean(traj_train))
    Par["out_shift"] = float(np.mean(traj_train))
    inp_std = float(np.std(traj_train))
    out_std = float(np.std(traj_train))
    Par["inp_scale"] = float(max(inp_std, 1e-6))
    Par["out_scale"] = float(max(out_std, 1e-6))
    Par["t_shift"] = float(t_min)
    Par["t_scale"] = float(t_max - t_min)
    Par["time_cond"] = time_cond.tolist()

    # 物理损失消融：仅将两个物理项权重置 0，不影响其他逻辑
    Par["mse_weight"] = 1.0
    Par["div_weight"] = 0.0  # Ablation: disable divergence loss
    Par["momentum_weight"] = 0.0  # Ablation: disable momentum loss
    logger.info(
        f"[Ablation] Physical weights - MSE: {Par['mse_weight']}, Div: {Par['div_weight']}, Momentum: {Par['momentum_weight']}\n"
    )

    Par["mask"] = paddle.to_tensor(mask, dtype=data_type)

    def convert_to_serializable(obj):
        if isinstance(obj, (np.ndarray, paddle.Tensor)):
            return obj.tolist()
        return obj

    with open(os.path.join(main_output_dir, "Par.json"), "w") as f:
        json.dump(Par, f, default=convert_to_serializable)

    # 预处理与数据集（保持一致）
    traj_train_tensor = paddle.to_tensor(data=traj_train, dtype=data_type)
    traj_val_tensor = paddle.to_tensor(data=traj_val, dtype=data_type)
    traj_test_tensor = paddle.to_tensor(data=traj_test, dtype=data_type)
    time_cond_tensor = paddle.to_tensor(data=time_cond, dtype=data_type)
    begin_time = time.time()
    x_idx_train, t_idx_train, y_idx_train = preprocess_train(traj_train, Par)
    logger.info("Shape of train dataset")
    logger.info(f"x_idx_train: {x_idx_train.shape}")
    logger.info(f"t_idx_train: {t_idx_train.shape}")
    logger.info(f"y_idx_train: {y_idx_train.shape}\n")
    x_idx_val, t_idx_val, y_idx_val = preprocess(traj_val, Par)
    logger.info("Shape of val dataset")
    logger.info(f"x_idx_val: {x_idx_val.shape}")
    logger.info(f"t_idx_val: {t_idx_val.shape}")
    logger.info(f"y_idx_val: {y_idx_val.shape}\n")
    x_idx_test, t_idx_test, y_idx_test = preprocess(traj_test, Par)
    logger.info("Shape of test dataset")
    logger.info(f"x_idx_test: {x_idx_test.shape}")
    logger.info(f"t_idx_test: {t_idx_test.shape}")
    logger.info(f"y_idx_test: {y_idx_test.shape}\n")
    logger.info(f"Data preprocess time: {time.time() - begin_time:.2f}s\n")

    train_dataset = YourDataset_train(x_idx_train, t_idx_train, y_idx_train)
    val_dataset = YourDataset(x_idx_val, y_idx_val)
    test_dataset = YourDataset(x_idx_test, y_idx_test)

    train_batch_size = 4
    val_batch_size = 4
    test_batch_size = 4
    logger.info(
        f"Batch size of train, val, and test: {train_batch_size}, {val_batch_size}, {test_batch_size}"
    )
    train_loader = paddle.io.DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        use_shared_memory=False,
    )
    val_loader = paddle.io.DataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        num_workers=max(0, int(args.num_workers)),
        use_shared_memory=False,
    )
    test_loader = paddle.io.DataLoader(
        dataset=test_dataset,
        batch_size=test_batch_size,
        num_workers=max(0, int(args.num_workers)),
        use_shared_memory=False,
    )

    # 模型（保持与基准一致）
    model = Unet2D_with_FNO_without_atte_FCRB(
        dim=16,
        Par=Par,
        dim_mults=(1, 2, 4, 8),
        channels=Par["channels"],
        attention_heads=None,
    ).astype("float32")

    # 损失函数（与基准一致，权重来自 Par）
    criterion = CustomLoss(Par, mask=Par.get("mask"))

    # 优化器与调度器（保持一致）
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=1e-4,
        weight_decay=1e-6,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=1.0),
    )
    scheduler = combined_scheduler(
        optimizer,
        Par["num_epochs"] * len(train_loader),
        int(0.1 * Par["num_epochs"]) * len(train_loader),
    )

    # 训练循环（与基准一致）
    num_epochs = Par["num_epochs"]
    PLOT_EVERY = max(0, int(args.plot_every))
    best_val_loss = float("inf")
    best_model_id = 0
    train_losses = []
    val_losses = []

    models_dir = os.path.join(main_output_dir, "models")
    images_dir = os.path.join(main_output_dir, "images")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    loss_csv_path = os.path.join(main_output_dir, "loss_curve.csv")
    try:
        with open(loss_csv_path, "w", encoding="utf8") as f:
            f.write("epoch,train_loss,val_loss\n")
    except Exception as e:
        logger.warning(f"Failed to init loss CSV: {e}")

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        begin_time = time.time()
        model.train()
        train_loss = 0.0
        batch_count = 0

        # 物理项 warmup（保持与基准一致，权重已置零不会影响数值）
        phys_warmup_epochs = max(30, int(0.2 * num_epochs))
        warmup_factor = min(1.0, epoch / float(phys_warmup_epochs))
        try:
            criterion.set_warmup_factor(warmup_factor)
        except Exception:
            pass

        for x_idx, t_idx, y_idx in tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
            disable=not sys.stdout.isatty(),
        ):
            x = traj_train_tensor[0, x_idx]
            t = time_cond_tensor[t_idx]
            y_true = traj_train_tensor[0, y_idx]
            optimizer.clear_gradients(set_to_zero=False)

            with paddle.amp.auto_cast(enable=False):
                y_pred = model(x, t)
            loss = criterion(y_pred.astype("float32"), y_true.astype("float32"))

            if not np.isfinite(float(loss.item())):
                y_pred_safe = paddle.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
                try:
                    criterion.set_warmup_factor(0.0)
                except Exception:
                    pass
                loss_fallback = criterion(y_pred_safe.astype("float32"), y_true.astype("float32"))
                if not np.isfinite(float(loss_fallback.item())):
                    logger.warning("Non-finite loss detected; skipping this batch")
                    optimizer.clear_gradients(set_to_zero=True)
                    continue
                else:
                    loss = loss_fallback

            loss.backward()
            try:
                grads = []
                for p in model.parameters():
                    if p.grad is not None:
                        grads.append(paddle.flatten(p.grad))
                if grads:
                    total_norm = float(paddle.linalg.norm(paddle.concat(grads), p=2).numpy())
                    if not np.isfinite(total_norm):
                        logger.warning("Non-finite grad norm detected; skipping optimizer update for this batch")
                        optimizer.clear_gradients(set_to_zero=True)
                        continue
                    grad_log_every = max(0, int(args.grad_log_every))
                    grad_log_epochs = max(0, int(args.grad_log_epochs))
                    if (
                        grad_log_every > 0
                        and epoch <= grad_log_epochs
                        and (batch_count % grad_log_every == 0)
                    ):
                        logger.info(f"Grad norm: {total_norm:.3e}")
            except Exception:
                pass

            optimizer.step()
            train_loss += loss.item()
            scheduler.step()
            batch_count += 1

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # 验证（与基准一致）
        model.eval()
        val_loss = 0.0
        with paddle.no_grad():
            for x_idx, y_idx in val_loader:
                x = traj_val_tensor[0, x_idx]
                t = time_cond_tensor[t_idx_val]
                y_true = traj_val_tensor[0, y_idx]
                NT = Par["lb"] + y_true.shape[1]
                y_pred = rollout(model, x, t, NT, Par, val_batch_size)
                loss = criterion(y_pred.astype("float32"), y_true.astype("float32"))
                val_loss += loss.item()

            val_loss_avg = val_loss / len(val_loader) if len(val_loader) > 0 else float("nan")
            val_losses.append(val_loss_avg)
            if PLOT_EVERY and ((epoch == 1) or (epoch == num_epochs) or (epoch % PLOT_EVERY == 0)):
                make_plot(
                    y_true.detach().cpu().numpy(),
                    y_pred.detach().cpu().numpy(),
                    epoch,
                    images_dir,
                    train_losses=train_losses,
                    val_losses=val_losses,
                )
        val_loss = val_loss_avg
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_id = epoch
            paddle.save(obj=model.state_dict(), path=os.path.join(models_dir, "best_model.pdparams"))
        elapsed_time = time.time() - begin_time
        logger.info(
            f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4e}, Val Loss: {val_loss:.4e}, "
            f"Best model: {best_model_id}, Learning rate: {scheduler.get_lr():.4e}, "
            f"Epoch time: {elapsed_time:.2f}"
        )
        try:
            with open(loss_csv_path, "a", encoding="utf8") as f:
                f.write(f"{epoch},{train_loss:.6e},{val_loss:.6e}\n")
        except Exception as e:
            logger.warning(f"Failed to append loss CSV: {e}")

    logger.info("Training finished.")
    logger.info(f"Training Time: {time.time() - t0:.1f}s")

    # 测试阶段（与基准一致）
    model.eval()
    test_loss = 0.06
    with paddle.no_grad():
        for x_idx, y_idx in test_loader:
            x = traj_test_tensor[0, x_idx]
            t = time_cond_tensor[t_idx_test]
            y_true = traj_test_tensor[0, y_idx]
            NT = Par["lb"] + y_true.shape[1]
            y_pred = rollout(model, x, t, NT, Par, val_batch_size)
            loss = criterion(y_pred.astype("float32"), y_true.astype("float32"))
            test_loss += loss.item()
    test_loss /= len(test_loader)
    logger.info(f"Test Loss: {test_loss:.4e}")
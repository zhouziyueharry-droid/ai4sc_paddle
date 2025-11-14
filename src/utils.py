import json
import logging
import math
import os
import time
import subprocess
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paddle
import scipy.stats as stats
from numpy.lib.stride_tricks import sliding_window_view


def setup_seed(seed):
    paddle.seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    # 在 CPU 环境或部分 Paddle 版本中，以下标志不可通过 set_flags 设置。
    # 为兼容性，改为条件设置或通过环境变量提示，避免报错。
    try:
        if hasattr(paddle, "is_compiled_with_cuda") and paddle.is_compiled_with_cuda():
            # 优先通过环境变量提示保持确定性，避免 set_flags 抛错
            os.environ.setdefault("FLAGS_cudnn_deterministic", "True")
        os.environ.setdefault("FLAGS_benchmark", "False")
    except Exception:
        pass


def init_all(seed, name, dtype):
    setup_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        paddle.set_default_dtype(dtype)
    except Exception:
        # 兼容某些版本的 Paddle，若设置失败则忽略
        pass

    def select_best_gpu():
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            best_idx = 0
            best_score = float("inf")
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                score = (mem.used / max(mem.total, 1)) + (util / 100.0) * 0.5
                if score < best_score:
                    best_score = score
                    best_idx = i
            pynvml.nvmlShutdown()
            return best_idx
        except Exception:
            pass
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
                errors="ignore",
            )
            best_idx = 0
            best_score = float("inf")
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    i = int(parts[0])
                    util = float(parts[1])
                    mem_used = float(parts[2])
                    mem_total = float(parts[3])
                    score = (mem_used / max(mem_total, 1)) + (util / 100.0) * 0.5
                    if score < best_score:
                        best_score = score
                        best_idx = i
            return best_idx
        except Exception:
            pass
        return 0

    try:
        gpu_available = bool(paddle.is_compiled_with_cuda())
        try:
            gpu_count = paddle.device.cuda.device_count()
        except Exception:
            gpu_count = 0
        if gpu_available and gpu_count > 0:
            chosen_idx = select_best_gpu()
            device = f"gpu:{chosen_idx}"
        else:
            device = "cpu"
        paddle.device.set_device(device)
    except Exception as e:
        device = f"cpu (fallback: {e})"

    if not os.path.exists(name):
        os.makedirs(name)
    log_level = logging.INFO
    log_name = os.path.join(name, time.strftime("%Y-%m-%d-%H-%M-%S") + ".log")
    logger = logging.getLogger("")
    logger.setLevel(log_level)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_name, encoding="utf8")
    file_handler.setLevel(level=log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Project name: {name}")
    logger.info(f"Random seed value: {seed}, data type : {dtype}")
    logger.info(f"Device: {device}")
    logger.info(f"CUDA available: {gpu_available}, GPU count: {gpu_count}\n")
    return logger


def make_plot(TRUE, PRED, epoch, images_dir="images", train_losses=None, val_losses=None, val_rollout_losses=None, val_1step_losses=None, mask2d=None, zero_center=True, unify_scale=False):
    sample_id = 0
    T = TRUE.shape[1]
    skip_t = max(1, T // 8)
    idx_all = np.arange(T)[::skip_t]
    N = min(idx_all.shape[0], 3)
    idx = idx_all[:N]

    true = np.nan_to_num(TRUE[sample_id, idx, 0], nan=0.0, posinf=0.0, neginf=0.0)
    pred1 = np.nan_to_num(PRED[sample_id, idx, 0], nan=0.0, posinf=0.0, neginf=0.0)
    true_uy = np.nan_to_num(TRUE[sample_id, idx, 1], nan=0.0, posinf=0.0, neginf=0.0)
    pred_uy = np.nan_to_num(PRED[sample_id, idx, 1], nan=0.0, posinf=0.0, neginf=0.0)
    time_ls = idx
    CMAP = "turbo"

    def rel_l2(Ta, Pa):
        try:
            dtype_str = paddle.get_default_dtype()
            if not isinstance(dtype_str, str):
                dtype_str = 'float32'
        except Exception:
            dtype_str = 'float32'
        Tt = paddle.to_tensor(data=Ta, dtype=dtype_str)
        Pt = paddle.to_tensor(data=Pa, dtype=dtype_str)
        numer = paddle.linalg.norm(x=Tt - Pt, p=2)
        denom = paddle.linalg.norm(x=Tt, p=2)
        val = numer / paddle.clip(denom, min=1e-8)
        return float(val.item())

    def frame_vmin_vmax(x):
        vals = np.asarray(x).flatten()
        vals = vals[np.isfinite(vals)]
        vals = vals[np.abs(vals) > 1e-8]
        if vals.size < 50:
            vals = np.asarray(x).flatten()
            vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vmin = float(np.nanmin(x))
            vmax = float(np.nanmax(x))
            return vmin, vmax
        vmin = np.percentile(vals, 1)
        vmax = np.percentile(vals, 99)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            m = float(vals.mean())
            s = float(vals.std())
            if s > 0:
                vmin = m - 3.0 * s
                vmax = m + 3.0 * s
            else:
                vmin = float(vals.min())
                vmax = float(vals.max())
        if vmin == vmax:
            eps = 1e-12 if vmax == 0 else abs(vmax) * 1e-6
            vmin -= eps
            vmax += eps
        return float(vmin), float(vmax)

    fig, axes = plt.subplots(6, N, figsize=(30, 14))
    axes = np.array(axes)
    cbar_ax = fig.add_axes([0.92, 0.26, 0.02, 0.62])

    m2d = None
    if mask2d is not None:
        m2d = np.asarray(mask2d)
        if m2d.ndim > 2:
            m2d = np.squeeze(m2d)
        m2d = m2d.astype(np.float32)

    kbins = None
    kvals = None
    for i in range(N):
        t_frame = true[i]
        p_frame = pred1[i]
        if m2d is not None:
            t_frame = t_frame * m2d
            p_frame = p_frame * m2d
        vmin_t, vmax_t = frame_vmin_vmax(t_frame)
        vmin_p, vmax_p = frame_vmin_vmax(p_frame)
        if zero_center:
            max_abs_t = max(abs(vmin_t), abs(vmax_t))
            max_abs_p = max(abs(vmin_p), abs(vmax_p))
            max_abs_t = max(max_abs_t, 1e-12)
            max_abs_p = max(max_abs_p, 1e-12)
            vmin_t, vmax_t = -max_abs_t, max_abs_t
            vmin_p, vmax_p = -max_abs_p, max_abs_p
        if unify_scale:
            max_abs_pair = max(abs(vmin_t), abs(vmax_t), abs(vmin_p), abs(vmax_p))
            max_abs_pair = max(max_abs_pair, 1e-12)
            vmin_t = vmin_p = -max_abs_pair
            vmax_t = vmax_p = max_abs_pair
        im = axes[0, i].imshow(t_frame, vmin=vmin_t, vmax=vmax_t, cmap=CMAP)
        axes[0, i].set_title(f"Time: {int(time_ls[i])}s (Ux)", fontsize=14)
        axes[0, i].axis("off")

        im = axes[1, i].imshow(p_frame, vmin=vmin_p, vmax=vmax_p, cmap=CMAP)
        mse_val1 = rel_l2(t_frame, p_frame)
        axes[1, i].set_title(f"rel L2 (Ux): {mse_val1:.2e}", fontsize=12)
        axes[1, i].axis("off")

        image = true[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        kbins = np.arange(0.5, min(nx, ny) // 2 + 1, 1.0)
        kvals = 0.5 * (kbins[1:] + kbins[:-1])
        Abins_true, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins_true *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_true = np.maximum(Abins_true, 1e-12)
        axes[2, i].loglog(kvals, Abins_true, 'b-', label="True Ux", linewidth=2)

        image = pred1[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        Abins_pred, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins_pred *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_pred = np.maximum(Abins_pred, 1e-12)
        axes[2, i].loglog(kvals, Abins_pred, 'r--', label="Pred Ux", linewidth=2)

        k_ref = np.linspace(1, np.max(kbins), 100)
        energy_ref = k_ref ** (-5 / 3)
        energy_ref *= max(max(Abins_true), max(Abins_pred)) / max(energy_ref)
        axes[2, i].loglog(k_ref, energy_ref, "k:", label="k^-5/3", alpha=0.7)
        axes[2, i].legend(fontsize=9)
        axes[2, i].set_xlabel("$k$")
        axes[2, i].set_title("Ux Energy Spectrum", fontsize=12)

        ty_frame = true_uy[i]
        py_frame = pred_uy[i]
        if m2d is not None:
            ty_frame = ty_frame * m2d
            py_frame = py_frame * m2d
        vmin_ty, vmax_ty = frame_vmin_vmax(ty_frame)
        vmin_py, vmax_py = frame_vmin_vmax(py_frame)
        if zero_center:
            max_abs_ty = max(abs(vmin_ty), abs(vmax_ty))
            max_abs_py = max(abs(vmin_py), abs(vmax_py))
            max_abs_ty = max(max_abs_ty, 1e-12)
            max_abs_py = max(max_abs_py, 1e-12)
            vmin_ty, vmax_ty = -max_abs_ty, max_abs_ty
            vmin_py, vmax_py = -max_abs_py, max_abs_py
        if unify_scale:
            max_abs_pair_y = max(abs(vmin_ty), abs(vmax_ty), abs(vmin_py), abs(vmax_py))
            max_abs_pair_y = max(max_abs_pair_y, 1e-12)
            vmin_ty = vmin_py = -max_abs_pair_y
            vmax_ty = vmax_py = max_abs_pair_y
        im2 = axes[3, i].imshow(ty_frame, vmin=vmin_ty, vmax=vmax_ty, cmap=CMAP)
        axes[3, i].set_title(f"Time: {int(time_ls[i])}s (Uy)", fontsize=14)
        axes[3, i].axis("off")

        im2 = axes[4, i].imshow(py_frame, vmin=vmin_py, vmax=vmax_py, cmap=CMAP)
        mse_val_u = rel_l2(ty_frame, py_frame)
        axes[4, i].set_title(f"rel L2 (Uy): {mse_val_u:.2e}", fontsize=12)
        axes[4, i].axis("off")

        image = true_uy[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        Abins_true, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins_true *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_true = np.maximum(Abins_true, 1e-12)

        image = pred_uy[i]
        ny, nx = image.shape
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image) ** 2
        kfreq_y = np.fft.fftfreq(ny) * ny
        kfreq_x = np.fft.fftfreq(nx) * nx
        kfreq2D_x, kfreq2D_y = np.meshgrid(kfreq_x, kfreq_y)
        knrm = np.sqrt(kfreq2D_x**2 + kfreq2D_y**2).flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()
        Abins_pred, _, _ = stats.binned_statistic(knrm, fourier_amplitudes, statistic="mean", bins=kbins)
        Abins_pred *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
        Abins_pred = np.maximum(Abins_pred, 1e-12)

        axes[5, i].loglog(kvals, Abins_true, label="Simulated Uy", color="tab:blue")
        axes[5, i].loglog(kvals, Abins_pred, label="MATCHO Uy", color="tab:orange")
        k_ref = np.linspace(1, np.max(kbins), 100)
        energy_ref = k_ref ** (-5 / 3)
        energy_ref *= max(max(Abins_true), max(Abins_pred)) / max(energy_ref)
        axes[5, i].loglog(k_ref, energy_ref, "k--", label="k^-5/3")
        axes[5, i].legend(fontsize=10)
        axes[5, i].set_title(f"Uy Energy Spectrum (t={int(time_ls[i])})")
        axes[5, i].set_xlabel("$k$")
        axes[5, i].set_ylabel("Energy")

    fig.subplots_adjust(left=0.06, right=0.9, top=0.96, bottom=0.22, hspace=0.8, wspace=0.25)
    if (val_rollout_losses is None) and (val_losses is not None):
        val_rollout_losses = val_losses
    if train_losses is not None and val_rollout_losses is not None:
        try:
            loss_ax = fig.add_axes([0.08, 0.04, 0.8, 0.14])
            x = np.arange(1, len(train_losses) + 1)
            val_x = np.arange(1, len(val_rollout_losses) + 1)
            train_arr = np.asarray(train_losses, dtype=float)
            val_arr = np.asarray(val_rollout_losses, dtype=float)
            val1_arr = np.asarray(val_1step_losses, dtype=float) if val_1step_losses is not None else None
            train_arr = np.where(np.isfinite(train_arr), train_arr, np.nan)
            val_arr = np.where(np.isfinite(val_arr), val_arr, np.nan)
            if val1_arr is not None:
                val1_arr = np.where(np.isfinite(val1_arr), val1_arr, np.nan)

            has_train_pos = np.any((train_arr > 0) & np.isfinite(train_arr))
            has_val_pos = np.any((val_arr > 0) & np.isfinite(val_arr))
            has_val1_pos = np.any((val1_arr > 0) & np.isfinite(val1_arr)) if val1_arr is not None else False
            use_log = bool(has_train_pos or has_val_pos or has_val1_pos)

            if use_log:
                loss_ax.set_yscale("log")
                train_plot = np.where(train_arr > 0, train_arr, np.nan)
                val_plot = np.where(val_arr > 0, val_arr, np.nan)
                loss_ax.plot(x, train_plot, label="Train loss", color="tab:blue", linewidth=2)
                loss_ax.plot(val_x, val_plot, label="Val rollout loss", color="tab:orange", linestyle="--", marker="o", markersize=3, alpha=0.9, linewidth=2, zorder=3)
                if val1_arr is not None:
                    val1_x = np.arange(1, len(val1_arr) + 1)
                    val1_plot = np.where(val1_arr > 0, val1_arr, np.nan)
                    loss_ax.plot(val1_x, val1_plot, label="Val 1-step loss", color="tab:green", linestyle=":", marker="s", markersize=3, alpha=0.9, linewidth=2, zorder=2)
                pos_vals = np.concatenate([
                    train_arr[(train_arr > 0) & np.isfinite(train_arr)],
                    val_arr[(val_arr > 0) & np.isfinite(val_arr)],
                    val1_arr[(val1_arr > 0) & np.isfinite(val1_arr)] if val1_arr is not None else np.array([], dtype=float),
                ])
                if pos_vals.size > 0:
                    y_min = float(np.nanmin(pos_vals))
                    y_max = float(np.nanmax(pos_vals))
                    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > 0:
                        loss_ax.set_ylim(max(y_min, 1e-12), y_max * 1.2)
                loss_ax.set_ylabel("Loss (log)")
            else:
                loss_ax.set_yscale("linear")
                loss_ax.plot(x, train_arr, label="Train loss", color="tab:blue", linewidth=2)
                loss_ax.plot(val_x, val_arr, label="Val rollout loss", color="tab:orange", linestyle="--", marker="o", markersize=3, alpha=0.9, linewidth=2, zorder=3)
                if val1_arr is not None:
                    val1_x = np.arange(1, len(val1_arr) + 1)
                    loss_ax.plot(val1_x, val1_arr, label="Val 1-step loss", color="tab:green", linestyle=":", marker="s", markersize=3, alpha=0.9, linewidth=2, zorder=2)
                loss_ax.set_ylabel("Loss")

            loss_ax.set_xlabel("Epoch")
            loss_ax.set_title(f"Loss curves through epoch {epoch + 1}")
            loss_ax.grid(True, which="both", ls=":", alpha=0.3)
            loss_ax.legend(loc="upper right")
        except Exception:
            pass

    fig.colorbar(im, cax=cbar_ax)
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, f"{epoch + 1}.png"))
    plt.close(fig)


class CustomLoss(paddle.nn.Layer):
    def __init__(self, Par, mask=None):
        super(CustomLoss, self).__init__()
        self.Par = Par
        self.mse_weight = Par.get('mse_weight', 1.0)
        self.div_weight = Par.get('div_weight', 0.1)
        self.momentum_weight = Par.get('momentum_weight', 0.05)
        self._warmup = 1.0
        self.Ma = 0.3
        self.Re = 23000
        self.nu = 1.0 / self.Re
        self.mask = mask

    def set_warmup_factor(self, factor: float):
        try:
            factor = float(factor)
        except Exception:
            factor = 1.0
        self._warmup = max(0.0, min(1.0, factor))

    def _to_channel_last(self, tensor):
        shape = list(tensor.shape)
        if len(shape) == 0:
            return tensor
        if shape[-1] == 2:
            return tensor
        if len(shape) == 4 and shape[1] == 2:
            return paddle.transpose(tensor, perm=[0, 2, 3, 1])
        if len(shape) == 5 and shape[2] == 2:
            return paddle.transpose(tensor, perm=[0, 1, 3, 4, 2])
        if 2 in shape:
            c_axis = shape.index(2)
            perm = [i for i in range(len(shape)) if i != c_axis] + [c_axis]
            return paddle.transpose(tensor, perm=perm)
        return tensor

    def _ddx(self, u):
        left = u[..., :, 1:2] - u[..., :, 0:1]
        center = (u[..., :, 2:] - u[..., :, :-2]) * 0.5
        right = u[..., :, -1:] - u[..., :, -2:-1]
        return paddle.concat([left, center, right], axis=-1)

    def _ddy(self, u):
        top = u[..., 1:2, :] - u[..., 0:1, :]
        center = (u[..., 2:, :] - u[..., :-2, :]) * 0.5
        bottom = u[..., -1:, :] - u[..., -2:-1, :]
        return paddle.concat([top, center, bottom], axis=-2)

    def _d2dx2(self, u):
        left = u[..., :, 1:2] - 2.0 * u[..., :, 0:1] + u[..., :, 0:1]
        center = u[..., :, 2:] - 2.0 * u[..., :, 1:-1] + u[..., :, :-2]
        right = u[..., :, -1:] - 2.0 * u[..., :, -1:] + u[..., :, -2:-1]
        return paddle.concat([left, center, right], axis=-1)

    def _d2dy2(self, u):
        top = u[..., 1:2, :] - 2.0 * u[..., 0:1, :] + u[..., 0:1, :]
        center = u[..., 2:, :] - 2.0 * u[..., 1:-1, :] + u[..., :-2, :]
        bottom = u[..., -1:, :] - 2.0 * u[..., -1:, :] + u[..., -2:-1, :]
        return paddle.concat([top, center, bottom], axis=-2)

    def forward(self, y_pred, y_true):
        y_true = (y_true - self.Par["out_shift"]) / (self.Par["out_scale"])
        y_pred = (y_pred - self.Par["out_shift"]) / (self.Par["out_scale"])
        y_true = paddle.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = paddle.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
        y_true = self._to_channel_last(y_true)
        y_pred = self._to_channel_last(y_pred)
        if self.mask is not None:
            mask_tensor = paddle.to_tensor(self.mask, dtype=y_pred.dtype)
            while len(mask_tensor.shape) < len(y_pred.shape) - 1:
                mask_tensor = paddle.unsqueeze(mask_tensor, axis=0)
            mask_tensor = paddle.unsqueeze(mask_tensor, axis=-1)
            squared_diff = (y_true - y_pred) ** 2
            masked_squared_diff = squared_diff * mask_tensor
            denom = paddle.sum(mask_tensor) * float(y_pred.shape[-1])
            denom = paddle.clip(denom, min=1e-12)
            mse_loss = paddle.sum(masked_squared_diff) / denom
        else:
            mse_loss = paddle.mean((y_true - y_pred) ** 2)
        physics_loss = paddle.to_tensor(0.0, dtype=y_pred.dtype)
        if self.div_weight > 0:
            ux = y_pred[..., 0]
            uy = y_pred[..., 1]
            try:
                ux = paddle.clip(ux, min=-10.0, max=10.0)
                uy = paddle.clip(uy, min=-10.0, max=10.0)
            except Exception:
                pass
            divergence = self._ddx(ux) + self._ddy(uy)
            if self.mask is not None:
                mask_tensor = paddle.to_tensor(self.mask, dtype=divergence.dtype)
                while len(mask_tensor.shape) < len(divergence.shape):
                    mask_tensor = paddle.unsqueeze(mask_tensor, axis=0)
                div_loss = paddle.sum((divergence ** 2) * mask_tensor) / paddle.sum(mask_tensor)
            else:
                div_loss = paddle.mean(divergence ** 2)
            physics_loss = physics_loss + (self.div_weight * self._warmup) * div_loss
        if self.momentum_weight > 0:
            ux = y_pred[..., 0]
            uy = y_pred[..., 1]
            try:
                ux = paddle.clip(ux, min=-10.0, max=10.0)
                uy = paddle.clip(uy, min=-10.0, max=10.0)
            except Exception:
                pass
            dudx = self._ddx(ux)
            dudy = self._ddy(ux)
            dvdx = self._ddx(uy)
            dvdy = self._ddy(uy)
            d2udx2 = self._d2dx2(ux)
            d2udy2 = self._d2dy2(ux)
            d2vdx2 = self._d2dx2(uy)
            d2vdy2 = self._d2dy2(uy)
            convective_x = ux * dudx + uy * dudy
            convective_y = ux * dvdx + uy * dvdy
            viscous_x = self.nu * (d2udx2 + d2udy2)
            viscous_y = self.nu * (d2vdx2 + d2vdy2)
            momentum_residual_x = convective_x - viscous_x
            momentum_residual_y = convective_y - viscous_y
            momentum_residual = momentum_residual_x ** 2 + momentum_residual_y ** 2
            if self.mask is not None:
                mask_tensor = paddle.to_tensor(self.mask, dtype=momentum_residual.dtype)
                while len(mask_tensor.shape) < len(momentum_residual.shape):
                    mask_tensor = paddle.unsqueeze(mask_tensor, axis=0)
                momentum_loss = paddle.sum(momentum_residual * mask_tensor) / paddle.sum(mask_tensor)
            else:
                momentum_loss = paddle.mean(momentum_residual)
            physics_loss = physics_loss + (self.momentum_weight * self._warmup) * momentum_loss
        total_loss = self.mse_weight * mse_loss + physics_loss
        if hasattr(self, 'loss_components'):
            try:
                self.loss_components = {
                    'mse_loss': mse_loss.numpy(),
                    'physics_loss': physics_loss.numpy(),
                }
            except Exception:
                pass
        return total_loss


class YourDataset_train(paddle.io.Dataset):
    def __init__(self, x, t, y, transform=None):
        self.x = x
        self.t = t
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        t_sample = self.t[idx]
        y_sample = self.y[idx]
        if self.transform:
            x_sample, t_sample, y_sample = self.transform(x_sample, t_sample, y_sample)
        return x_sample, t_sample, y_sample


class YourDataset(paddle.io.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x_sample = self.x[idx]
        y_sample = self.y[idx]
        if self.transform:
            x_sample, y_sample = self.transform(x_sample, y_sample)
        return x_sample, y_sample


def preprocess_train(traj, Par):
    nt = traj.shape[1]
    temp = nt - Par["lb"] - Par["lf"] + 1
    x_idx = np.arange(temp).reshape(-1, 1)
    x_idx = np.tile(x_idx, (1, Par["lf"]))
    x_idx = x_idx.reshape(-1, 1)
    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx + i)
    x_idx = np.concatenate(x_idx_ls, axis=1)
    t_idx = np.arange(Par["lf"]).reshape(1, -1)
    t_idx = np.tile(t_idx, (temp, 1)).reshape(-1)
    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par["lb"] :], window_shape=Par["lf"]).reshape(-1)
    return (
        paddle.to_tensor(data=x_idx, dtype="int64"),
        paddle.to_tensor(data=t_idx, dtype="int64"),
        paddle.to_tensor(data=y_idx, dtype="int64"),
    )


def preprocess(traj, Par):
    nt = traj.shape[1]
    if nt - Par["lb"] < 1:
        raise ValueError(f"序列过短：nt={nt}, lb={Par['lb']}，至少需要 nt - lb >= 1")
    effective_LF = min(Par["LF"], nt - Par["lb"])  # 自适应窗口长度
    temp = nt - Par["lb"] - effective_LF + 1
    x_idx = np.arange(temp).reshape(-1, 1)
    x_idx_ls = []
    for i in range(Par["lb"]):
        x_idx_ls.append(x_idx + i)
    x_idx = np.concatenate(x_idx_ls, axis=1)
    t_idx = np.arange(Par["lf"]).reshape(-1)
    y_idx = np.arange(nt)
    y_idx = sliding_window_view(y_idx[Par["lb"] :], window_shape=effective_LF)
    return (
        paddle.to_tensor(data=x_idx, dtype="int64"),
        paddle.to_tensor(data=t_idx, dtype="int64"),
        paddle.to_tensor(data=y_idx, dtype="int64"),
    )


def combined_scheduler(optimizer, total_epochs, warmup_epochs, last_epoch=-1):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    tmp_lr = paddle.optimizer.lr.LambdaDecay(
        lr_lambda=lr_lambda, last_epoch=last_epoch, learning_rate=optimizer.get_lr()
    )
    optimizer.set_lr_scheduler(tmp_lr)
    return tmp_lr


def rollout(model, x, t, NT, Par, batch_size):
    y_pred_ls = []
    bs = batch_size
    end = bs

    while True:
        start = end - bs
        out_ls = []

        if start >= x.shape[0]:
            break
        temp_x1 = x[start:end]
        out_ls = [temp_x1]
        traj = paddle.concat(x=out_ls, axis=1)

        while traj.shape[1] < NT:
            with paddle.no_grad():
                temp_x = paddle.repeat_interleave(x=temp_x1, repeats=Par["lf"], axis=0)
                temp_t = t.tile(repeat_times=traj.shape[0])
                out = model(temp_x, temp_t).reshape([-1, Par["lf"], Par["nf"], Par["nx"], Par["ny"]])
                out_ls.append(out)
                traj = paddle.concat(x=out_ls, axis=1)
                temp_x1 = traj[:, -Par["lb"] :]
        pred = paddle.concat(x=out_ls, axis=1)[:, Par["lb"] : NT]
        y_pred_ls.append(pred)
        end = end + bs
        if end - bs > x.shape[0] + 1:
            break

    if len(y_pred_ls) > 0:
        y_pred = paddle.concat(y_pred_ls, axis=0)
    else:
        y_pred = paddle.zeros([0, NT - Par["lb"], Par["nf"], Par["nx"], Par["ny"]])
    return y_pred
"""
评估脚本（FCRB 版本）：与原版 evaluate_test.py 内容一致，唯一差异是模型导入使用
`Unet2D_with_FNO_without_atte_FCRB`（通过别名保持构造代码不变）。

使用方式与原版相同，建议通过命令行传入 `--train_dir` 指向 FCRB 训练输出主目录。
"""

# --- 兼容 CLI 环境的小型引导：保证像 IDE 一样可导入 src，并优先使用 conda 运行时库 ---
import os as _os
import sys as _sys
import pathlib as _pl
_proj_root = _pl.Path(__file__).resolve().parents[1]
if str(_proj_root) not in _sys.path:
    _sys.path.insert(0, str(_proj_root))
_conda_prefix = _os.environ.get("CONDA_PREFIX")
if _conda_prefix:
    if _os.name == "nt":
        # Windows: 为 DLL 搜索路径添加 conda 的库目录
        try:
            _os.add_dll_directory(_pl.Path(_conda_prefix) / "Library" / "bin")
        except Exception:
            pass
    else:
        # Linux: 将 conda 的 lib 提前到 LD_LIBRARY_PATH，优先加载新版本运行时
        _os.environ["LD_LIBRARY_PATH"] = f"{_conda_prefix}/lib:" + _os.environ.get("LD_LIBRARY_PATH", "")

import argparse
import json
import os
import time
import numpy as np
import paddle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys, pathlib
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# 与训练脚本保持一致：使用 Final.model.unet_withFNO 中的模型与 Final.train_utils 中的工具
from src.model import Unet2D_with_FNO_without_atte_FCRB
from src.utils import preprocess, rollout, CustomLoss, make_plot

# ------------- 新增：基础数值与物理工具函数 -------------
def rmse_value(true, pred):
    """Root Mean Square Error（RMSE）。"""
    diff = true - pred
    return float(np.sqrt(np.mean(diff * diff)))


def rel_rmse_value(true, pred):
    """相对 RMSE：RMSE / ||true||。"""
    rms_true = float(np.sqrt(np.mean(true * true)))
    return rmse_value(true, pred) / (rms_true + 1e-8)


def linf_value(true, pred):
    """L∞ 误差（最大绝对误差）。"""
    return float(np.max(np.abs(true - pred)))


def ssim_global(img1, img2, C1=1e-4, C2=9e-4):
    """简化版 SSIM（全局统计近似），避免额外依赖。"""
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = x.var()
    sigma_y2 = y.var()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2)
    return float(num / (den + 1e-12))


def compute_vorticity_divergence(u, v):
    """计算二维速度场的涡度与散度（二阶差分边界）。"""
    du_dx = np.gradient(u, axis=0, edge_order=2)
    du_dy = np.gradient(u, axis=1, edge_order=2)
    dv_dx = np.gradient(v, axis=0, edge_order=2)
    dv_dy = np.gradient(v, axis=1, edge_order=2)
    omega = dv_dx - du_dy
    div = du_dx + dv_dy
    return omega, div


def energy_enstrophy(u, v):
    """计算能量密度与涡度平方（Enstrophy）的均值。"""
    E = 0.5 * (u * u + v * v)
    omega, _ = compute_vorticity_divergence(u, v)
    enst = 0.5 * (omega * omega)
    return float(E.mean()), float(enst.mean())


def energy_spectrum_2d(field):
    """计算二维场的各向同性能谱 E(k)。返回 k_bins, E_k。"""
    fhat = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(fhat) ** 2
    nx, ny = field.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
    ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    kmax = int(np.max(K))
    E_k = []
    k_bins = []
    for k in range(kmax + 1):
        mask = (K >= k) & (K < k + 1)
        if np.any(mask):
            E_k.append(float(power[mask].mean()))
            k_bins.append(k + 0.5)
    return np.asarray(k_bins, dtype=np.float32), np.asarray(E_k, dtype=np.float32)


def setup_device():
    """选择并设置计算设备。

    优先使用 GPU（若可用且存在设备），否则回退到 CPU。
    返回字符串："gpu" 或 "cpu"。
    """
    try:
        gpu_available = bool(paddle.is_compiled_with_cuda())
        try:
            gpu_count = paddle.device.cuda.device_count()
        except Exception:
            gpu_count = 0
        device = "gpu" if gpu_available and gpu_count > 0 else "cpu"
        paddle.device.set_device(device)
    except Exception:
        device = "cpu"
    return device


def rel_l2_np(true, pred):
    """相对 L2 误差：||true - pred|| / (||true|| + eps)。"""
    numer = np.linalg.norm(true - pred)
    denom = np.linalg.norm(true)
    return float(numer / (denom + 1e-8))


def mae_np(true, pred):
    """平均绝对误差（MAE）。"""
    return float(np.mean(np.abs(true - pred)))


def mse_np(true, pred):
    """均方误差（MSE）。"""
    diff = true - pred
    return float(np.mean(diff * diff))


def make_plot_simple(TRUE, PRED, epoch, images_dir="images_test"):
    """占位简图（保留接口），当前主可视化在评估流程中生成。"""
    os.makedirs(images_dir, exist_ok=True)
    try:
        make_plot(TRUE, PRED, epoch, images_dir)
    except Exception:
        pass


def evaluate(ux_path, uy_path, mask_path, train_dir=None, par_json_path=None, model_path=None, output_dir=None, batch_size=10, LF=None):
    """运行评估流程并生成新版测试报告与可视化。

    参数
    - ux_path/uy_path/mask_path: 预处理后的 `UX.npy`/`UY.npy`/`mask.npy` 路径。
    - train_dir: 训练输出目录（用于自动查找 `Par.json` 与最佳模型权重）。
    - par_json_path/model_path: 可显式指定配置与权重路径，优先级高于 `train_dir` 推断。
    - output_dir: 评估输出目录（曲线、CSV、图像、预测样本等）。
    - batch_size: 推理与评估时的窗口批大小。
    - LF: 替换/覆盖长预测步数（若不提供则使用配置中的值或默认）。

    说明：本函数严格保持原有数值与管线逻辑，仅补充文档说明。
    """
    os.makedirs(output_dir, exist_ok=True)
    device = setup_device()
    print(f"Device: {device}")

    # 解析 Par.json 与模型路径（优先 train_dir 推断，支持 train_dir 为子目录如 test_report 的回退）
    candidates = []
    if train_dir:
        candidates.append(train_dir)
        parent_dir = os.path.dirname(train_dir.rstrip("/\\"))
        if parent_dir and parent_dir != train_dir:
            candidates.append(parent_dir)

    # 自动推断 Par.json（先查 train_dir，再查其父目录）
    if par_json_path is None:
        for base in candidates:
            cand = os.path.join(base, "Par.json")
            if os.path.isfile(cand):
                par_json_path = cand
                break
    # 自动推断模型权重路径（同样的查找顺序）
    if model_path is None:
        for base in candidates:
            cand = os.path.join(base, "models", "best_model.pdparams")
            if os.path.isfile(cand):
                model_path = cand
                break
    # 校验并给出详细报错（会列出尝试过的候选路径）
    errors = []
    if par_json_path is None or not os.path.isfile(par_json_path):
        errors.append(
            f"Par.json 未找到，尝试路径: {[os.path.join(b, 'Par.json') for b in candidates] or ['<未提供 train_dir>']}"
        )
    if model_path is None or not os.path.isfile(model_path):
        errors.append(
            f"模型未找到，尝试路径: {[os.path.join(b, 'models', 'best_model.pdparams') for b in candidates] or ['<未提供 train_dir>']}"
        )
    if errors:
        raise FileNotFoundError("; ".join(errors))

    # 加载数据（保持与训练版一致的维度与通道组织）
    traj_ux = np.load(ux_path)
    traj_uy = np.load(uy_path)
    traj_ux = np.expand_dims(traj_ux, axis=0)
    traj_uy = np.expand_dims(traj_uy, axis=0)
    if traj_ux.shape != traj_uy.shape:
        raise ValueError(f"Ux 与 Uy 形状不一致：{traj_ux.shape} vs {traj_uy.shape}")
    traj = np.stack([traj_ux, traj_uy], axis=2)  # [N, T, C, nx, ny]

    # 掩膜广播与对齐（支持 [nx,ny]、[B,nx,ny]、[B,T,nx,ny] 等形状）
    mask_np = np.load(mask_path)
    mask_np = mask_np.astype(np.float32)
    if mask_np.ndim == 2:
        mask_np = mask_np[None, None, ...]
    elif mask_np.ndim == 3:
        mask_np = mask_np[:, None, ...]
    elif mask_np.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected mask shape: {mask_np.shape}")
    traj = traj * mask_np

    # 集合划分：80% 训练、10% 验证、10% 测试（此处仅取测试段）
    nt_all = traj.shape[1]
    train_end = int(nt_all * 0.8)
    val_end = int(nt_all * 0.9)
    train_end = max(1, min(train_end, nt_all - 2))
    val_end = max(train_end + 1, min(val_end, nt_all - 1))
    traj_test = traj[:, val_end:]

    # 加载 Par 并校准关键字段（nx/ny/nf 与通道数、时间相关超参）
    with open(par_json_path, "r", encoding="utf8") as f:
        Par = json.load(f)
    Par["nx"] = int(traj_test.shape[-2])
    Par["ny"] = int(traj_test.shape[-1])
    Par["nf"] = int(2)
    Par["lb"] = int(Par.get("lb", 10))
    Par["lf"] = int(Par.get("lf", 2))

    # 评估预测窗口长度：优先使用命令行传入的 LF，否则保留 Par 中已有值或默认 100
    if LF is not None:
        try:
            Par["LF"] = int(LF)
        except Exception:
            Par["LF"] = int(Par.get("LF", 100))
    else:
        Par["LF"] = int(Par.get("LF", 100))
    Par["channels"] = int(Par.get("channels", Par["nf"] * Par["lb"]))
    time_cond = np.asarray(Par.get("time_cond", np.linspace(0, 1, Par["lf"])), dtype=np.float32)

    # 在 Par 中保留掩膜张量（训练中用于损失与约束）
    Par["mask"] = paddle.to_tensor(mask_np, dtype="float32")  # [1,1,nx,ny]

    # 构建张量与索引（调用训练版 preprocess/rollout 保持一致窗口/时序）
    traj_test_tensor = paddle.to_tensor(data=traj_test, dtype="float32")
    time_cond_tensor = paddle.to_tensor(data=time_cond, dtype="float32")
    x_idx_test, t_idx_test, y_idx_test = preprocess(traj_test, Par)

    # 模型与权重（与训练脚本完全一致的构造参数）
    model = Unet2D_with_FNO_without_atte_FCRB(
        dim=16,
        Par=Par,
        dim_mults=(1, 2, 4, 8),
        channels=Par["channels"],
        attention_heads=None,
    ).astype("float32")
    state_dict = paddle.load(model_path)
    model.set_state_dict(state_dict)
    model.eval()

    # -------- 模型推理速度衡量（输入 lb 帧 → 输出 lf 帧 的一次前向）--------
    speed_metrics = {}
    try:
        with paddle.no_grad():
            # 选取少量窗口做基准（不改变后续评估逻辑）
            total_windows = int(x_idx_test.shape[0])
            speed_batch_size = max(1, min(batch_size, total_windows))
            temp_x1 = traj_test_tensor[0, x_idx_test[:speed_batch_size]]  # [B, lb, nf, nx, ny]
            t_vec = time_cond_tensor[t_idx_test]                           # [lf]
            temp_x = paddle.repeat_interleave(x=temp_x1, repeats=Par["lf"], axis=0)  # [B*lf, lb, nf, nx, ny]
            temp_t = t_vec.tile(repeat_times=temp_x1.shape[0])                         # [B*lf]

            lat = []
            warmup = 5
            # 可选同步（GPU 下保证计时准确；CPU 下忽略）
            def _sync():
                try:
                    paddle.device.cuda.synchronize()
                except Exception:
                    pass

            # warmup
            for _ in range(warmup):
                _sync(); _t0 = time.perf_counter(); _ = model(temp_x, temp_t); _sync(); _t1 = time.perf_counter()
            # measure 10 次
            for _ in range(10):
                _sync(); t0 = time.perf_counter(); _ = model(temp_x, temp_t); _sync(); t1 = time.perf_counter()
                lat.append((t1 - t0) * 1000.0)  # ms
            lat_arr = np.asarray(lat, dtype=np.float64)
            speed_metrics = {
                "lb": int(Par["lb"]),
                "lf": int(Par["lf"]),
                "batch_size_measure": int(speed_batch_size),
                "forward_mean_ms": float(lat_arr.mean()),
                "forward_p50_ms": float(np.median(lat_arr)),
                "forward_p95_ms": float(np.percentile(lat_arr, 95)),
                "per_sample_mean_ms": float(lat_arr.mean() / max(1, speed_batch_size)),
                "throughput_samples_per_s": float( (speed_batch_size) / (lat_arr.mean() / 1000.0) ),
            }
    except Exception:
        pass

    # 评估与新版指标收集
    criterion = CustomLoss(Par, mask=Par.get("mask"))
    images_dir = os.path.join(output_dir, "images_test")
    os.makedirs(images_dir, exist_ok=True)
    csv_frame = os.path.join(output_dir, "metrics_frame.csv")
    csv_phys = os.path.join(output_dir, "metrics_physics.csv")
    summary_json = os.path.join(output_dir, "summary_test.json")
    preds_path = os.path.join(output_dir, "pred_test.npy")

    test_loss = 0.0
    all_pred_samples = []
    rmse_heat_rows = []
    begin = time.time()
    with paddle.no_grad():
        total_windows = x_idx_test.shape[0]
        per_tau_rmse = None
        per_tau_relrmse = None
        per_tau_mae = None
        per_tau_linf = None
        per_tau_ssim_mag = None
        per_tau_vort_rmse = None
        per_tau_div_mean = None
        per_tau_div_rmse = None
        per_tau_energy_true = None
        per_tau_energy_pred = None
        per_tau_enst_true = None
        per_tau_enst_pred = None
        per_tau_skill = None

        for start in range(0, total_windows, batch_size):
            end = min(start + batch_size, total_windows)
            x = traj_test_tensor[0, x_idx_test[start:end]]
            t = time_cond_tensor[t_idx_test]
            y_true = traj_test_tensor[0, y_idx_test[start:end]]
            NT = Par["lb"] + y_true.shape[1]
            y_pred = rollout(model, x, t, NT, Par, batch_size)
            loss = criterion(y_pred.astype("float32"), y_true.astype("float32")).item()
            test_loss += loss

            y_true_np = y_true.numpy()
            y_pred_np = y_pred.numpy()
            if len(all_pred_samples) < 1:
                all_pred_samples.append((y_true_np, y_pred_np))

            B, Tm, C, nx, ny = y_true_np.shape
            if per_tau_rmse is None:
                per_tau_rmse = [[] for _ in range(Tm)]
                per_tau_relrmse = [[] for _ in range(Tm)]
                per_tau_mae = [[] for _ in range(Tm)]
                per_tau_linf = [[] for _ in range(Tm)]
                per_tau_ssim_mag = [[] for _ in range(Tm)]
                per_tau_vort_rmse = [[] for _ in range(Tm)]
                per_tau_div_mean = [[] for _ in range(Tm)]
                per_tau_div_rmse = [[] for _ in range(Tm)]
                per_tau_energy_true = [[] for _ in range(Tm)]
                per_tau_energy_pred = [[] for _ in range(Tm)]
                per_tau_enst_true = [[] for _ in range(Tm)]
                per_tau_enst_pred = [[] for _ in range(Tm)]
                per_tau_skill = [[] for _ in range(Tm)]

            x_last = x.numpy()[:, -1]
            y_base = np.repeat(x_last[:, None, ...], Tm, axis=1)

            batch_heat = [[] for _ in range(B)]
            for tstep in range(Tm):
                true_t = y_true_np[:, tstep]
                pred_t = y_pred_np[:, tstep]
                base_t = y_base[:, tstep]
                for b in range(B):
                    u_true = true_t[b, 0]
                    v_true = true_t[b, 1]
                    u_pred = pred_t[b, 0]
                    v_pred = pred_t[b, 1]
                    u_base = base_t[b, 0]
                    v_base = base_t[b, 1]

                    mag_true = np.sqrt(u_true * u_true + v_true * v_true)
                    mag_pred = np.sqrt(u_pred * u_pred + v_pred * v_pred)
                    mag_base = np.sqrt(u_base * u_base + v_base * v_base)

                    rmse_val = rmse_value(mag_true, mag_pred)
                    relrmse_val = rel_rmse_value(mag_true, mag_pred)
                    mae_val = mae_np(mag_true, mag_pred)
                    linf_val = linf_value(mag_true, mag_pred)
                    ssim_val = ssim_global(mag_true, mag_pred)

                    per_tau_rmse[tstep].append(rmse_val)
                    per_tau_relrmse[tstep].append(relrmse_val)
                    per_tau_mae[tstep].append(mae_val)
                    per_tau_linf[tstep].append(linf_val)
                    per_tau_ssim_mag[tstep].append(ssim_val)

                    omega_true, div_true = compute_vorticity_divergence(u_true, v_true)
                    omega_pred, div_pred = compute_vorticity_divergence(u_pred, v_pred)
                    vort_rmse = rmse_value(omega_true, omega_pred)
                    div_mean_true = float(div_true.mean())
                    div_rmse = rmse_value(div_true, div_pred)
                    E_true_mean, enst_true_mean = energy_enstrophy(u_true, v_true)
                    E_pred_mean, enst_pred_mean = energy_enstrophy(u_pred, v_pred)

                    per_tau_vort_rmse[tstep].append(vort_rmse)
                    per_tau_div_mean[tstep].append(div_mean_true)
                    per_tau_div_rmse[tstep].append(div_rmse)
                    per_tau_energy_true[tstep].append(E_true_mean)
                    per_tau_energy_pred[tstep].append(E_pred_mean)
                    per_tau_enst_true[tstep].append(enst_true_mean)
                    per_tau_enst_pred[tstep].append(enst_pred_mean)

                    rmse_model = rmse_val
                    rmse_baseline = rmse_value(mag_true, mag_base)
                    skill = 1.0 - (rmse_model / (rmse_baseline + 1e-8))
                    per_tau_skill[tstep].append(skill)

                    batch_heat[b].append(rmse_val)

            rmse_heat_rows.extend(batch_heat)
            last_chunk = y_pred_np

    test_loss /= max(1, (total_windows + batch_size - 1) // batch_size)

    def stats(arr):
        a = np.asarray(arr, dtype=np.float64)
        return float(a.mean()), float(np.median(a)), float(a.std(ddof=0))

    with open(csv_frame, "w", encoding="utf8") as f:
        f.write(
            "tau,RMSE_mean,RMSE_median,RMSE_std,relRMSE_mean,relRMSE_median,relRMSE_std,MAE_mean,MAE_median,MAE_std,Linf_mean,Linf_median,Linf_std,SSIM_mag_mean,SSIM_mag_median,SSIM_mag_std,Skill_mean,Skill_median,Skill_std\n"
        )
        for tstep, (
            r_list,
            rr_list,
            m_list,
            lf_list,
            s_list,
            sk_list,
        ) in enumerate(zip(
            per_tau_rmse,
            per_tau_relrmse,
            per_tau_mae,
            per_tau_linf,
            per_tau_ssim_mag,
            per_tau_skill,
        )):
            rm = stats(r_list)
            rrm = stats(rr_list)
            ma = stats(m_list)
            li = stats(lf_list)
            ss = stats(s_list)
            sk = stats(sk_list)
            f.write(
                f"{tstep+1},{rm[0]:.4g},{rm[1]:.4g},{rm[2]:.4g},{rrm[0]:.4g},{rrm[1]:.4g},{rrm[2]:.4g},{ma[0]:.4g},{ma[1]:.4g},{ma[2]:.4g},{li[0]:.4g},{li[1]:.4g},{li[2]:.4g},{ss[0]:.4g},{ss[1]:.4g},{ss[2]:.4g},{sk[0]:.4g},{sk[1]:.4g},{sk[2]:.4g}\n"
            )

    with open(csv_phys, "w", encoding="utf8") as f:
        f.write(
            "tau,Vorticity_RMSE_mean,Vorticity_RMSE_median,Vorticity_RMSE_std,Div_mean_mean,Div_mean_median,Div_mean_std,Div_RMSE_mean,Div_RMSE_median,Div_RMSE_std,Energy_true_mean,Energy_pred_mean,Energy_error_mean,Enstrophy_true_mean,Enstrophy_pred_mean\n"
        )
        for tstep in range(len(per_tau_vort_rmse)):
            vr = stats(per_tau_vort_rmse[tstep])
            dm = stats(per_tau_div_mean[tstep])
            dr = stats(per_tau_div_rmse[tstep])
            Et = stats(per_tau_energy_true[tstep])[0]
            Ep = stats(per_tau_energy_pred[tstep])[0]
            Eerr = Ep - Et
            En_t = stats(per_tau_enst_true[tstep])[0]
            En_p = stats(per_tau_enst_pred[tstep])[0]
            f.write(
                f"{tstep+1},{vr[0]:.4g},{vr[1]:.4g},{vr[2]:.4g},{dm[0]:.4g},{dm[1]:.4g},{dm[2]:.4g},{dr[0]:.4g},{dr[1]:.4g},{dr[2]:.4g},{Et:.4g},{Ep:.4g},{Eerr:.4g},{En_t:.4g},{En_p:.4g}\n"
            )

    taus = np.arange(1, len(per_tau_rmse) + 1)
    rmse_mean = [np.mean(v) for v in per_tau_rmse]
    mae_mean = [np.mean(v) for v in per_tau_mae]
    skill_mean = [np.mean(v) for v in per_tau_skill]
    plt.figure(figsize=(10, 6))
    plt.plot(taus, rmse_mean, label="RMSE (magnitude)")
    plt.plot(taus, mae_mean, label="MAE (magnitude)")
    plt.plot(taus, skill_mean, label="Skill")
    plt.xlabel("lead time τ")
    plt.ylabel("metric value")
    plt.title("Errors and skill vs lead time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(images_dir, "curves_metrics.png"))
    plt.close()

    vort_rmse_mean = [np.mean(v) for v in per_tau_vort_rmse]
    enst_true_mean_curve = [np.mean(v) for v in per_tau_enst_true]
    enst_pred_mean_curve = [np.mean(v) for v in per_tau_enst_pred]
    plt.figure(figsize=(10, 6))
    plt.plot(taus, vort_rmse_mean, label="Vorticity RMSE")
    plt.plot(taus, enst_true_mean_curve, label="Enstrophy True")
    plt.plot(taus, enst_pred_mean_curve, label="Enstrophy Pred")
    plt.xlabel("lead time τ")
    plt.ylabel("value")
    plt.title("Physical metrics vs lead time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(images_dir, "curves_physics.png"))
    plt.close()

    if rmse_heat_rows:
        heat = np.asarray(rmse_heat_rows, dtype=np.float32)
        plt.figure(figsize=(12, 6))
        plt.imshow(heat, aspect="auto", cmap="turbo")
        plt.colorbar(label="RMSE (magnitude)")
        plt.xlabel("lead time τ")
        plt.ylabel("samples")
        plt.title("Error heatmap")
        plt.savefig(os.path.join(images_dir, "error_heatmap.png"))
        plt.close()

    selected_taus = [1, 2, 5, 10]
    for tau in selected_taus:
        tidx = tau - 1
        if tidx < 0 or tidx >= len(per_tau_rmse):
            continue
        if all_pred_samples:
            y_true_sample, y_pred_sample = all_pred_samples[0]
            B = y_true_sample.shape[0]
            k_bins_list_true = []
            Ek_list_true = []
            k_bins_list_pred = []
            Ek_list_pred = []
            for b in range(B):
                u_true = y_true_sample[b, tidx, 0]
                v_true = y_true_sample[b, tidx, 1]
                u_pred = y_pred_sample[b, tidx, 0]
                v_pred = y_pred_sample[b, tidx, 1]
                mag_true = np.sqrt(u_true * u_true + v_true * v_true)
                mag_pred = np.sqrt(u_pred * u_pred + v_pred * v_pred)
                k_true, Ek_true = energy_spectrum_2d(mag_true)
                k_pred, Ek_pred = energy_spectrum_2d(mag_pred)
                if len(k_true) and len(k_pred):
                    k_bins_list_true.append(k_true)
                    Ek_list_true.append(Ek_true)
                    k_bins_list_pred.append(k_pred)
                    Ek_list_pred.append(Ek_pred)
            if k_bins_list_true and k_bins_list_pred:
                k_len = min(min(len(k) for k in k_bins_list_true), min(len(k) for k in k_bins_list_pred))
                k_axis = k_bins_list_true[0][:k_len]
                Ek_true_mean = np.mean([Ek[:k_len] for Ek in Ek_list_true], axis=0)
                Ek_pred_mean = np.mean([Ek[:k_len] for Ek in Ek_list_pred], axis=0)
                dEk = (Ek_pred_mean - Ek_true_mean) / (Ek_true_mean + 1e-12)
                np.savez(os.path.join(output_dir, f"spectrum_tau_{tau}.npz"), k=k_axis, E_true=Ek_true_mean, E_pred=Ek_pred_mean, dE=dEk)
                plt.figure(figsize=(8, 6))
                plt.loglog(k_axis, Ek_true_mean + 1e-12, label="True")
                plt.loglog(k_axis, Ek_pred_mean + 1e-12, label="Pred")
                plt.xlabel("k")
                plt.ylabel("E(k)")
                plt.title(f"Energy spectrum comparison (tau={tau})")
                plt.legend()
                plt.grid(True, which="both", alpha=0.3)
                plt.savefig(os.path.join(images_dir, f"spectrum_tau_{tau}.png"))
                plt.close()

    if all_pred_samples:
        y_true_sample, y_pred_sample = all_pred_samples[0]
        Tm = y_true_sample.shape[1]
        show_taus = [1, 2, 5, 10]
        for tau in show_taus:
            tidx = tau - 1
            if tidx < 0 or tidx >= Tm:
                continue
            u_true = y_true_sample[0, tidx, 0]
            v_true = y_true_sample[0, tidx, 1]
            u_pred = y_pred_sample[0, tidx, 0]
            v_pred = y_pred_sample[0, tidx, 1]
            omega_true, _ = compute_vorticity_divergence(u_true, v_true)
            omega_pred, _ = compute_vorticity_divergence(u_pred, v_pred)
            omega_err = omega_pred - omega_true
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            im0 = axes[0].imshow(omega_true, cmap="turbo")
            axes[0].set_title("Vorticity True")
            axes[0].axis("off")
            im1 = axes[1].imshow(omega_pred, cmap="turbo")
            axes[1].set_title("Vorticity Pred")
            axes[1].axis("off")
            im2 = axes[2].imshow(omega_err, cmap="turbo")
            axes[2].set_title("Vorticity Error")
            axes[2].axis("off")
            fig.colorbar(im0, ax=axes[0])
            fig.colorbar(im1, ax=axes[1])
            fig.colorbar(im2, ax=axes[2])
            plt.suptitle(f"Vorticity visualization (tau={tau})")
            plt.savefig(os.path.join(images_dir, f"vorticity_tau_{tau}.png"))
            plt.close(fig)

    # 新增：保存3个case的单张速度图（每case 40张：Ux True/Pred, Uy True/Pred × 前10帧）与竖向颜色标尺图
    if all_pred_samples:
        y_true_sample, y_pred_sample = all_pred_samples[0]
        B, Tm, C, nx, ny = y_true_sample.shape
        parent_dir = os.path.join(images_dir, "case_velocity_grids")
        os.makedirs(parent_dir, exist_ok=True)

        n_cases = int(min(3, B))
        n_cols = int(min(10, Tm))

        # 计算全局色标范围（覆盖选取的case与前n_cols帧，true+pred）
        u_max_all = 0.0
        v_max_all = 0.0
        for b_idx in range(n_cases):
            u_stack = np.concatenate([y_true_sample[b_idx, :n_cols, 0], y_pred_sample[b_idx, :n_cols, 0]], axis=0)
            v_stack = np.concatenate([y_true_sample[b_idx, :n_cols, 1], y_pred_sample[b_idx, :n_cols, 1]], axis=0)
            u_max_all = max(u_max_all, float(np.max(np.abs(u_stack))))
            v_max_all = max(v_max_all, float(np.max(np.abs(v_stack))))

        # 每个case一个文件夹，内含40张单图
        # 统计残差的全局范围（Ux/Uy），用于残差色标参考图
        ru_max_all = 0.0
        rv_max_all = 0.0
        for b_idx in range(n_cases):
            case_dir = os.path.join(parent_dir, f"case_{b_idx}")
            os.makedirs(case_dir, exist_ok=True)
            for t in range(n_cols):
                # Ux True
                fig = plt.figure(figsize=(6, 5))
                plt.imshow(y_true_sample[b_idx, t, 0], cmap="turbo", vmin=-u_max_all, vmax=u_max_all)
                plt.axis("off")
                plt.savefig(os.path.join(case_dir, f"tau_{t+1}_Ux_true.png"), bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # Uy True
                fig = plt.figure(figsize=(6, 5))
                plt.imshow(y_true_sample[b_idx, t, 1], cmap="turbo", vmin=-v_max_all, vmax=v_max_all)
                plt.axis("off")
                plt.savefig(os.path.join(case_dir, f"tau_{t+1}_Uy_true.png"), bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # Ux Pred
                fig = plt.figure(figsize=(6, 5))
                plt.imshow(y_pred_sample[b_idx, t, 0], cmap="turbo", vmin=-u_max_all, vmax=u_max_all)
                plt.axis("off")
                plt.savefig(os.path.join(case_dir, f"tau_{t+1}_Ux_pred.png"), bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # Uy Pred
                fig = plt.figure(figsize=(6, 5))
                plt.imshow(y_pred_sample[b_idx, t, 1], cmap="turbo", vmin=-v_max_all, vmax=v_max_all)
                plt.axis("off")
                plt.savefig(os.path.join(case_dir, f"tau_{t+1}_Uy_pred.png"), bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # Residual Ux: pred - true
                res_u = y_pred_sample[b_idx, t, 0] - y_true_sample[b_idx, t, 0]
                ru_max = float(np.max(np.abs(res_u))) if hasattr(np, "max") else np.max(np.abs(res_u))
                ru_max_all = max(ru_max_all, ru_max)
                fig = plt.figure(figsize=(6, 5))
                plt.imshow(res_u, cmap="seismic", vmin=-ru_max, vmax=ru_max)
                plt.axis("off")
                plt.savefig(os.path.join(case_dir, f"tau_{t+1}_Ux_residual.png"), bbox_inches="tight", pad_inches=0)
                plt.close(fig)

                # Residual Uy: pred - true
                res_v = y_pred_sample[b_idx, t, 1] - y_true_sample[b_idx, t, 1]
                rv_max = float(np.max(np.abs(res_v))) if hasattr(np, "max") else np.max(np.abs(res_v)
                              )
                rv_max_all = max(rv_max_all, rv_max)
                fig = plt.figure(figsize=(6, 5))
                plt.imshow(res_v, cmap="seismic", vmin=-rv_max, vmax=rv_max)
                plt.axis("off")
                plt.savefig(os.path.join(case_dir, f"tau_{t+1}_Uy_residual.png"), bbox_inches="tight", pad_inches=0)
                plt.close(fig)

        # 竖向颜色标尺参考图（全局范围，供绘图使用）
        try:
            from matplotlib import colors, cm
            cb_fig, cb_axes = plt.subplots(1, 2, figsize=(4.0, 8.0))
            norm_u = colors.Normalize(vmin=-u_max_all, vmax=u_max_all)
            norm_v = colors.Normalize(vmin=-v_max_all, vmax=v_max_all)
            cmap = plt.get_cmap("turbo")
            sm_u = cm.ScalarMappable(norm=norm_u, cmap=cmap)
            sm_v = cm.ScalarMappable(norm=norm_v, cmap=cmap)
            cb_u = plt.colorbar(sm_u, ax=cb_axes[0], orientation="vertical")
            cb_v = plt.colorbar(sm_v, ax=cb_axes[1], orientation="vertical")
            cb_axes[0].set_title("Ux colorbar")
            cb_axes[1].set_title("Uy colorbar")
            cb_u.set_label(f"Ux scale [-{u_max_all:.3g}, {u_max_all:.3g}]")
            cb_v.set_label(f"Uy scale [-{v_max_all:.3g}, {v_max_all:.3g}]")
            plt.suptitle("Vertical colorbar reference (turbo)")
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.savefig(os.path.join(parent_dir, "colorbar_reference_vertical.png"))
            plt.close(cb_fig)
        except Exception:
            pass

        # 残差竖向颜色标尺参考图（全局范围）
        try:
            from matplotlib import colors, cm
            rb_fig, rb_axes = plt.subplots(1, 2, figsize=(4.0, 8.0))
            norm_ru = colors.Normalize(vmin=-ru_max_all, vmax=ru_max_all)
            norm_rv = colors.Normalize(vmin=-rv_max_all, vmax=rv_max_all)
            cmap_res = plt.get_cmap("seismic")
            sm_ru = cm.ScalarMappable(norm=norm_ru, cmap=cmap_res)
            sm_rv = cm.ScalarMappable(norm=norm_rv, cmap=cmap_res)
            cb_ru = plt.colorbar(sm_ru, ax=rb_axes[0], orientation="vertical")
            cb_rv = plt.colorbar(sm_rv, ax=rb_axes[1], orientation="vertical")
            rb_axes[0].set_title("Ux residual")
            rb_axes[1].set_title("Uy residual")
            cb_ru.set_label(f"Ux residual [-{ru_max_all:.3g}, {ru_max_all:.3g}]")
            cb_rv.set_label(f"Uy residual [-{rv_max_all:.3g}, {rv_max_all:.3g}]")
            plt.suptitle("Vertical residual colorbar reference (seismic)")
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            plt.savefig(os.path.join(parent_dir, "colorbar_residual_vertical.png"))
            plt.close(rb_fig)
        except Exception:
            pass

    with open(summary_json, "w", encoding="utf8") as f:
        summary = {
            "test_loss": float(test_loss),
            "elapsed_sec": float(time.time() - begin),
            "RMSE_mean_first": float(rmse_mean[0]) if rmse_mean else None,
            "RMSE_mean_last": float(rmse_mean[-1]) if rmse_mean else None,
            "Skill_mean_first": float(skill_mean[0]) if skill_mean else None,
            "Skill_mean_last": float(skill_mean[-1]) if skill_mean else None,
            # 追加推理速度指标（一次前向：输入 lb 帧 → 输出 lf 帧）
            **({"speed": speed_metrics} if speed_metrics else {}),
        }
        json.dump(summary, f, indent=2)

    # 保存示例可视化与最后一块预测，便于报告与一致性检查
    if all_pred_samples:
        y_true_sample, y_pred_sample = all_pred_samples[0]
        make_plot_simple(y_true_sample, y_pred_sample, epoch=0, images_dir=images_dir)
    if 'last_chunk' in locals():
        np.save(preds_path, last_chunk)

    print(f"测试评估完成。报告目录：{output_dir}")


def main():
    """命令行入口：解析参数并触发评估。

    - 默认数据与 `DEFAULT_TRAIN_DIR` 仅用于占位，建议通过命令行显式传入。
    - 输出目录优先使用 `--output`，否则落到 `<train_dir>/test_report`。
    """
    # 与训练脚本一致的默认路径（可通过命令行覆盖）
    # 使用项目根目录下的 data 作为默认数据源（相对路径）
    DEFAULT_UX = str(project_root / "data" / "UX_nan_filtered.npy")
    DEFAULT_UY = str(project_root / "data" / "UY_nan_filtered.npy")
    DEFAULT_MASK = str(project_root / "data" / "mask.npy")
    # 如训练已完成，请将此目录改为对应的 train_results_YYYYMMDD_HHMMSS（主目录，而非其子目录）
    DEFAULT_TRAIN_DIR = r"/data/zhouziyue_benkesheng/AI4SC_program/Final/melting_train/train_results_20251108_164457"

    parser = argparse.ArgumentParser(description="Evaluate with advanced metrics (误差/物理/频谱/skill) [FCRB]")
    parser.add_argument("--ux", type=str, default=DEFAULT_UX, help="Ux .npy 路径")
    parser.add_argument("--uy", type=str, default=DEFAULT_UY, help="Uy .npy 路径")
    parser.add_argument("--mask", type=str, default=DEFAULT_MASK, help="掩膜 .npy 路径")
    parser.add_argument("--train_dir", type=str, default=DEFAULT_TRAIN_DIR, help="训练输出主目录（用于推断 Par.json 与模型路径）")
    parser.add_argument("--par", type=str, default=None, help="Par.json 路径（优先使用 train_dir 推断）")
    parser.add_argument("--model", type=str, default=None, help="模型 .pdparams 路径（优先使用 train_dir 推断）")
    parser.add_argument("--output", type=str, default=None, help="评估输出目录；不指定时默认为 <train_dir>/test_report")
    parser.add_argument("--batch_size", type=int, default=10, help="评估批大小，建议与训练中的验证/测试一致")
    parser.add_argument("--LF", type=int, default=20, help="评估预测窗口帧数（默认20）")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # 默认输出到项目根目录下的 test_output，以避免落到当前工作目录
    output_root_dir = project_root / "test_output"
    os.makedirs(output_root_dir, exist_ok=True)
    if args.output:
        output_dir = args.output
    else:
        output_dir = str(output_root_dir / ("test_results_" + timestamp))

    evaluate(
        ux_path=args.ux,
        uy_path=args.uy,
        mask_path=args.mask,
        train_dir=args.train_dir,
        par_json_path=args.par,
        model_path=args.model,
        output_dir=output_dir,
        batch_size=args.batch_size,
        LF=args.LF,
    )


if __name__ == "__main__":
    main()
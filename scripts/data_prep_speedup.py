"""
空气翼型 LES 原始数据预处理（加速版）。

- 保持原脚本的数值与流程逻辑不变（插值、掩膜、并行处理、保存结果等）。
- 仅规范代码格式，并补充少量说明性注释与文档字符串，便于后续维护与复现。
- 支持通过 CLI 参数配置输入/输出目录、目标网格大小与空间范围。
"""

import os
import pathlib
import argparse
from datetime import datetime

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import multiprocessing as mp

matplotlib.rcParams["figure.dpi"] = 200
plt.rcParams["font.family"] = "serif"


def interpolate_to_grid_with_mask(
    grid_x,
    grid_y,
    field,
    NX,
    NY,
    grid_xa=None,
    grid_ya=None,
    get_mask=False,
    x_range=None,
    y_range=None,
):
    """将散点数据线性插值到结构化网格，并可选生成机翼内部掩膜。

    保持与原脚本完全一致的处理逻辑：使用全局的 `X_RANGE`/`Y_RANGE` 来生成目标网格；
    当 `get_mask=True` 时，基于机翼轮廓点构造掩膜，机翼内部为 0，外部为 1。
    """
    # 保持与原脚本完全一致的处理逻辑（使用全局 X_RANGE/Y_RANGE）
    x_min, x_max = X_RANGE
    y_min, y_max = Y_RANGE
    target_x = np.linspace(x_min, x_max, NX)
    target_y = np.linspace(y_min, y_max, NY)
    grid_X, grid_Y = np.meshgrid(target_x, target_y)

    # 线性插值到结构化网格
    grid_Z = griddata((grid_x, grid_y), field, (grid_X, grid_Y), method="linear")

    if get_mask:
        airfoil_path = Path(np.column_stack((grid_xa, grid_ya)))
        points = np.vstack((grid_X.ravel(), grid_Y.ravel())).T
        inside_airfoil = airfoil_path.contains_points(points).reshape(NY, NX)
        mask = np.where(inside_airfoil, 0, 1)
    else:
        mask = None
    return grid_Z, mask


def fill_nan_with_nearest(data):
    """用就近有效值填充 `NaN` 或极端异常值。

    该函数通过距离变换找到每个 `NaN`/异常位置的最近有效索引，并用该位置的数据替换，
    以减少后续训练/评估过程中的数值问题。
    """
    nan_mask = np.isnan(data) | (np.abs(data) > 10)
    distances, indices = distance_transform_edt(nan_mask, return_indices=True)
    filled_data = data[tuple(indices)]
    data[nan_mask] = filled_data[nan_mask]
    return data


# 全局变量（由主进程或初始化函数设置）
GRID_X = None
GRID_Y = None
GRID_XA = None
GRID_YA = None
SCRIPT_DIR = None
RAW_DATA_DIR = None
NX = None
NY = None
skip_x = None
skip_y = None
X_RANGE = None
Y_RANGE = None
ON_ERROR = "fail"


def _init_worker(raw_data_dir, nx, ny, sx, sy, x_range, y_range):
    """每个进程的初始化：加载一次网格数据并设置全局参数。"""
    global GRID_X, GRID_Y, GRID_XA, GRID_YA, RAW_DATA_DIR, NX, NY, skip_x, skip_y, X_RANGE, Y_RANGE
    RAW_DATA_DIR = pathlib.Path(raw_data_dir).resolve()
    NX = nx
    NY = ny
    skip_x = sx
    skip_y = sy
    X_RANGE = x_range
    Y_RANGE = y_range

    # 加载网格数据（使用传入的原始数据目录）
    f = h5py.File(str(RAW_DATA_DIR / "airfoilLES_grid.h5"), "r")
    GRID_X = np.array(f["x"])
    GRID_Y = np.array(f["y"])
    GRID_XA = np.array(f["xa"])
    GRID_YA = np.array(f["ya"])
    f.close()


def _process_file(i):
    """处理单个时间步文件，返回下采样后的 ux/uy 网格以及第一帧的掩码。"""
    t_idx = str(100000 + i)[1:]
    raw_midspan_dir = RAW_DATA_DIR / "airfoilLES_midspan"
    path = str(raw_midspan_dir / f"airfoilLES_t{t_idx}.h5")
    print(f"reading: {path}")
    try:
        f = h5py.File(path, "r")
        ux = np.array(f["ux"])
        uy = np.array(f["uy"])
        f.close()
    except OSError as e:
        print(f"[Error] cannot open {path}: {e}")
        if ON_ERROR == "skip":
            return None, None, None
        raise

    # 与原脚本一致的插值及切片逻辑
    field_ux = ux
    if i == 1:
        grid_field_ux, grid_mask = interpolate_to_grid_with_mask(
            GRID_X, GRID_Y, field_ux, NX, NY, GRID_XA, GRID_YA, get_mask=True, x_range=X_RANGE, y_range=Y_RANGE
        )
        grid_mask_down = grid_mask[::skip_x, ::skip_y]
    else:
        grid_field_ux, _ = interpolate_to_grid_with_mask(
            GRID_X, GRID_Y, field_ux, NX, NY, x_range=X_RANGE, y_range=Y_RANGE
        )
        grid_mask_down = None

    temp_grid_field_ux = grid_field_ux[::skip_y, ::skip_x]

    field_uy = uy
    grid_field_uy, _ = interpolate_to_grid_with_mask(
        GRID_X, GRID_Y, field_uy, NX, NY, x_range=X_RANGE, y_range=Y_RANGE
    )
    temp_grid_field_uy = grid_field_uy[::skip_y, ::skip_x]

    return temp_grid_field_ux, temp_grid_field_uy, grid_mask_down


if __name__ == "__main__":
    # 主流程：解析参数、绘制网格示意、并行处理所有文件、保存结果（保持原逻辑不变）
    # 解析命令行参数以支持自定义输入/输出目录（默认使用项目根的 raw_data 与 data）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = pathlib.Path(script_dir).resolve().parent
    parser = argparse.ArgumentParser(description="Airfoil LES data preprocessing with configurable IO and grid params")
    parser.add_argument(
        "--input-dir", "-i", type=str, default=str(project_root / "raw_data"),
        help="原始数据目录，需包含 airfoilLES_grid.h5 与 airfoilLES_midspan 目录"
    )
    parser.add_argument(
        "--output-dir", "-o", type=str, default=str(project_root / "data"),
        help="输出目录，默认写入项目 data 目录"
    )
    # 网格与采样参数
    parser.add_argument("--nx", type=int, default=5 * 64, help="插值目标网格 X 方向点数，默认 5*64")
    parser.add_argument("--ny", type=int, default=8 * 64, help="插值目标网格 Y 方向点数，默认 8*64")
    parser.add_argument("--skip-x", type=int, default=2, help="X 方向下采样步长，默认 2")
    parser.add_argument("--skip-y", type=int, default=2, help="Y 方向下采样步长，默认 2")
    # 空间范围参数
    parser.add_argument("--x-min", type=float, default=-0.5, help="X 方向最小值，默认 -0.5")
    parser.add_argument("--x-max", type=float, default=2.0, help="X 方向最大值，默认 2.0")
    parser.add_argument("--y-min", type=float, default=-2.0, help="Y 方向最小值，默认 -2.0")
    parser.add_argument("--y-max", type=float, default=2.0, help="Y 方向最大值，默认 2.0")
    # 时间步索引范围
    parser.add_argument("--start-idx", type=int, default=1, help="起始时间步索引（包含），默认 1")
    parser.add_argument("--end-idx", type=int, default=1000, help="结束时间步索引（包含），默认 1000")
    parser.add_argument(
        "--on-error", type=str, choices=["fail", "skip"], default="fail",
        help="读取文件失败时的处理策略：fail 立即中止；skip 跳过坏文件继续"
    )
    parser.add_argument(
        "--validate-first", action="store_true",
        help="在并行处理前先预扫描并过滤坏的 HDF5 文件"
    )
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using input_dir: {input_dir}")
    print(f"Using output_dir: {output_dir}")
    if not (input_dir / "airfoilLES_grid.h5").exists():
        raise FileNotFoundError(f"airfoilLES_grid.h5 not found in {input_dir}")
    if not (input_dir / "airfoilLES_midspan").exists():
        raise FileNotFoundError(f"airfoilLES_midspan directory not found in {input_dir}")
    # 从命令行参数设置网格与范围
    NX = args.nx
    NY = args.ny
    skip_x = args.skip_x
    skip_y = args.skip_y
    X_RANGE = (args.x_min, args.x_max)
    Y_RANGE = (args.y_min, args.y_max)
    ON_ERROR = args.on_error
    print(NX,NY)

    print("Current directory:", script_dir)
    # 注：不再使用时间戳子目录，按参数指定目录输出（默认为项目 data）

    # 读取网格数据用于可视化（使用项目根目录 raw_data 相对路径）
    file = h5py.File(str(input_dir / "airfoilLES_grid.h5"), "r")
    grid_x = np.array(file["x"])
    grid_y = np.array(file["y"])
    grid_xa = np.array(file["xa"])
    grid_ya = np.array(file["ya"])
    file.close()

    plt.scatter(grid_x, grid_y, s=0.01, c="red")
    plt.scatter(grid_xa, grid_ya, s=1, c="blue")
    plt.savefig(str(output_dir / "grid.png"), dpi=600, bbox_inches="tight")

    # 并行处理所有时间步，保持顺序与首次掩码逻辑一致
    candidate_indices = list(range(args.start_idx, args.end_idx + 1))
    if args.validate_first:
        valid_indices = []
        invalid_indices = []
        mid_dir = input_dir / "airfoilLES_midspan"
        print("Validating HDF5 files before processing...")
        for i in tqdm(candidate_indices, desc="validate", leave=False):
            t_idx = str(100000 + i)[1:]
            path = mid_dir / f"airfoilLES_t{t_idx}.h5"
            try:
                # 快速验证：能够被 h5py 打开即认为有效
                f = h5py.File(str(path), "r")
                f.close()
                valid_indices.append(i)
            except Exception:
                invalid_indices.append(i)
        print(f"Validation done: valid={len(valid_indices)}, invalid={len(invalid_indices)}")
        if len(valid_indices) == 0:
            raise RuntimeError("No valid HDF5 files found. Please check your input directory.")
        file_index = valid_indices
    else:
        file_index = candidate_indices
    ux_ls = []
    uy_ls = []
    grid_mask = None

    # 进程池初始化加载网格数据（每个进程一次），避免在主进程传输大数组
    processes = max(1, (os.cpu_count() or 2) - 1)
    with mp.Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(str(input_dir), NX, NY, skip_x, skip_y, X_RANGE, Y_RANGE),
    ) as pool:
        for temp_ux, temp_uy, mask_down in tqdm(pool.imap(_process_file, file_index), total=len(file_index)):
            if temp_ux is None or temp_uy is None:
                # 跳过坏文件
                continue
            ux_ls.append(temp_ux)
            uy_ls.append(temp_uy)
            if grid_mask is None and mask_down is not None:
                grid_mask = mask_down

    if len(ux_ls) == 0 or len(uy_ls) == 0:
        raise RuntimeError("No valid files processed; please check input directory or set --on-error fail")
    UX = np.array(ux_ls)
    UY = np.array(uy_ls)
    np.save(str(output_dir / "UX.npy"), UX)
    np.save(str(output_dir / "UY.npy"), UY)
    np.save(str(output_dir / "mask.npy"), grid_mask)

    UX_nan_filtered = fill_nan_with_nearest(UX)
    UY_nan_filtered = fill_nan_with_nearest(UY)
    np.save(str(output_dir / "UX_nan_filtered.npy"), UX_nan_filtered)
    np.save(str(output_dir / "UY_nan_filtered.npy"), UY_nan_filtered)

    
"""
compute_mean_std.py
计算 LIDARDUSTX2 训练集的 [x, y, z, intensity] 均值和标准差
结果用于替换 WadsPointCloudDataset 的 mean / std 默认参数

注意：
  - 只统计训练集（split.train），不碰 valid / test，避免数据泄露
  - 采用 Welford 在线算法，逐帧累加，不需要把所有点一次性加载进内存
"""

import os
import glob
import numpy as np
import yaml
from tqdm import tqdm

# ── 配置区 ────────────────────────────────────────────────────────────────────
DATA_ROOT    = "/home/bbb/dataset/data/LIDARDUSTX2"
LABEL_CONFIG = "binary_dedust.yaml"
# ──────────────────────────────────────────────────────────────────────────────


def get_train_bin_files(data_root, label_config):
    """从 yaml 读取训练集序列号，返回所有训练帧的 .bin 路径列表"""
    with open(label_config, 'r') as f:
        cfg = yaml.safe_load(f)
    train_seqs = cfg['split']['train']

    bin_files = []
    for seq in train_seqs:
        vel_dir = os.path.join(data_root, "sequences", str(seq), "velodyne")
        files   = sorted(glob.glob(os.path.join(vel_dir, "*.bin")))
        bin_files.extend(files)

    print(f"训练集序列数：{len(train_seqs)}")
    print(f"训练集帧数：  {len(bin_files)}")
    return bin_files


def compute_mean_std_welford(bin_files):
    """
    用 Welford 在线算法逐帧计算全局 mean / std，内存友好。

    Welford 公式（每新增一个点 x_n）：
        delta  = x_n - mean_{n-1}
        mean_n = mean_{n-1} + delta / n
        M_n    = M_{n-1} + delta * (x_n - mean_n)   # 累计方差分子
        std    = sqrt(M_n / n)
    """
    # 4 个通道：x, y, z, intensity
    n     = np.zeros(4, dtype=np.float64)   # 累计点数（每通道独立，应对 NaN）
    mean  = np.zeros(4, dtype=np.float64)
    M     = np.zeros(4, dtype=np.float64)   # 累计方差分子

    for bin_path in tqdm(bin_files, desc="计算 mean/std"):
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        # 过滤 NaN / Inf（m1 传感器可能残留）
        valid = np.isfinite(pts).all(axis=1)
        pts   = pts[valid].astype(np.float64)

        if pts.shape[0] == 0:
            continue

        for c in range(4):
            col = pts[:, c]
            for x in col:
                n[c]    += 1
                delta    = x - mean[c]
                mean[c] += delta / n[c]
                M[c]    += delta * (x - mean[c])

    # 上面逐点循环在大数据集上太慢，改用向量化 Welford（批量更新）
    # 重置，用向量化版本重新算
    n    = np.zeros(4, dtype=np.float64)
    mean = np.zeros(4, dtype=np.float64)
    M    = np.zeros(4, dtype=np.float64)

    for bin_path in tqdm(bin_files, desc="计算 mean/std（向量化）"):
        pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        valid = np.isfinite(pts).all(axis=1)
        pts   = pts[valid].astype(np.float64)

        if pts.shape[0] == 0:
            continue

        m_new = pts.shape[0]

        # 批量 Welford 合并公式（合并两组统计量）
        # 组 A：当前累计 (n, mean, M)
        # 组 B：本帧数据 (m_new, mean_b, var_b)
        mean_b = pts.mean(axis=0)
        var_b  = pts.var(axis=0)          # 总体方差（ddof=0）
        M_b    = var_b * m_new

        delta  = mean_b - mean
        n_new  = n + m_new

        # 避免 n=0 时除零
        safe_n = np.where(n_new > 0, n_new, 1)

        M    = M + M_b + delta**2 * (n * m_new / safe_n)
        mean = (mean * n + mean_b * m_new) / safe_n
        n    = n_new

    std = np.sqrt(M / n)

    return mean, std, n


def main():
    print(f"数据根目录：{DATA_ROOT}")
    print(f"配置文件：  {LABEL_CONFIG}\n")

    bin_files = get_train_bin_files(DATA_ROOT, LABEL_CONFIG)
    if not bin_files:
        raise RuntimeError("未找到训练集 .bin 文件，请检查路径和 yaml 配置")

    mean, std, n = compute_mean_std_welford(bin_files)

    print("\n" + "=" * 55)
    print("统计结果（仅训练集）")
    print("=" * 55)
    print(f"总点数：{n.astype(np.int64)}")
    print(f"\n通道       x          y          z       intensity")
    print(f"mean  {mean[0]:10.6f} {mean[1]:10.6f} {mean[2]:10.6f} {mean[3]:10.6f}")
    print(f"std   {std[0]:10.6f}  {std[1]:10.6f}  {std[2]:10.6f}  {std[3]:10.6f}")

    print("\n── 请将以下内容替换 WadsPointCloudDataset 的默认参数 ──")
    mean_str = "[" + ", ".join(f"{v:.7f}" for v in mean) + "]"
    std_str  = "[" + ", ".join(f"{v:.7f}" for v in std)  + "]"
    print(f"mean={mean_str},")
    print(f"std= {std_str},")
    print("=" * 55)


if __name__ == "__main__":
    main()

"""
remove_duplicate_dust_by_seq.py
针对 LIDARDUSTX 数据集的去重 + 按序列号分组脚本

文件名格式：sequence_{seq_id}_{sensor}_{frame_idx}_{timestamp}.bin / .label / .txt
例：        sequence_0113_ls64_0041_1691129277_5957658.bin
                       ^^^^  ^^^^
                       序列号  传感器类型

输入结构（每个传感器一个子目录，子目录内 bin/label/txt 平铺混合）：
    LIDARDUSTX/
    ├── ls64/
    │   ├── sequence_0111_ls64_0001_....bin
    │   ├── sequence_0111_ls64_0001_....label
    │   ├── sequence_0111_ls64_0001_....txt   ← 3D 框标注，忽略
    │   └── ...
    ├── ls128/  ly150/  ly300/  m1/  ouster/
    └── ...

输出结构（序列目录名为 {seq_id}_{sensor}，保证跨传感器唯一）：
    LIDARDUSTX2/
    └── sequences/
        ├── 0111_ls64/    velodyne/*.bin  labels/*.label
        ├── 0111_ls128/   velodyne/*.bin  labels/*.label
        └── ...

关于 label dtype 自动推断：
    不同传感器 / 标注工具可能用不同字节宽度存储 label：
        int32  (4 字节/点) — 最常见
        uint8  (1 字节/点) — ouster 等结构化雷达常用
        int16  (2 字节/点) — 部分工具输出
    脚本先由 bin 文件推算点数，再按 label 文件大小自动匹配 dtype，
    无需手动配置，兼容所有传感器。
"""

import os
import re
import glob
import shutil
import numpy as np

# ── 配置区（按需修改）──────────────────────────────────────────────────────────
SRC_ROOT     = "/home/bbb/dataset/data/LIDARDUSTX"   # 包含各传感器子目录的根目录
DST_ROOT     = "/home/bbb/dataset/data/LIDARDUSTX2"  # 输出根目录

SENSOR_TYPES = ["ls64", "ls128", "ly150", "ly300", "m1", "ouster"]

DUST_LABEL   = 9       # 沙尘点类别 ID
# ──────────────────────────────────────────────────────────────────────────────


# ══ 文件名解析 ════════════════════════════════════════════════════════════════

_FNAME_RE = re.compile(r'^sequence_([0-9]+)_([A-Za-z0-9]+)_(.+)$')

def parse_filename(stem):
    """
    从文件名主干解析序列号。
    例：stem = "sequence_0113_ls64_0041_1691129277_5957658"
        → seq_id = "0113"
    返回 seq_id 字符串，解析失败返回 None。
    """
    m = _FNAME_RE.match(stem)
    return m.group(1) if m else None


# ══ 安全清空输出目录 ══════════════════════════════════════════════════════════

def clean_output_dir():
    if os.path.exists(DST_ROOT):
        print(f"[清理] 检测到旧输出目录 {DST_ROOT}，正在删除…")
        shutil.rmtree(DST_ROOT)
        print(f"[清理] 已删除旧目录。")
    os.makedirs(DST_ROOT, exist_ok=True)


# ══ I/O 工具 ══════════════════════════════════════════════════════════════════

def read_bin(filepath):
    """
    读取 .bin 点云，返回 (N, 4) float32 [x, y, z, intensity]。
    同时返回点数 N，供 read_label 推断 dtype 使用。
    """
    points = np.fromfile(filepath, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(
            f"bin 文件大小 {points.size} 不是 4 的倍数，"
            f"可能不是 (x,y,z,i) 格式：{filepath}"
        )
    return points.reshape(-1, 4)


def read_label(filepath, num_points):
    """
    读取 .label 文件，返回 (N,) int32。

    根据文件大小与点数自动推断存储 dtype：
        文件大小 == num_points × 4  → int32  (4 字节/点)
        文件大小 == num_points × 2  → int16  (2 字节/点)
        文件大小 == num_points × 1  → uint8  (1 字节/点)
    其他情况抛出异常，提示实际比例供排查。
    """
    file_size = os.path.getsize(filepath)

    if file_size == num_points * 4:
        return np.fromfile(filepath, dtype=np.int32)
    elif file_size == num_points * 2:
        return np.fromfile(filepath, dtype=np.int16).astype(np.int32)
    elif file_size == num_points:
        return np.fromfile(filepath, dtype=np.uint8).astype(np.int32)
    else:
        ratio = file_size / num_points if num_points > 0 else float('nan')
        raise ValueError(
            f"label 文件大小 {file_size} 字节与点数 {num_points} 不匹配 "
            f"(比例={ratio:.4f})，支持 ×1/×2/×4：{filepath}"
        )


def remove_nan_points(raw_data, labels):
    """去除含 NaN 的点，同步过滤标签。返回 (clean_data, clean_labels, n_removed)"""
    valid_mask = ~np.isnan(raw_data).any(axis=1)
    n_removed  = int((~valid_mask).sum())
    return raw_data[valid_mask], labels[valid_mask], n_removed


# ══ 步骤 1：扫描文件，按 {seq_id}_{sensor} 分组 ═══════════════════════════════

def collect_by_sequence():
    """
    逐传感器子目录扫描 .bin 文件，同名匹配 .label，
    以 {seq_id}_{sensor} 为 key 分组，
    返回 { seq_key: [(bin_path, label_path, sensor), ...] }，按 seq_key 升序。
    """
    seq_map   = {}
    n_total   = 0
    n_missing = 0
    n_unknown = 0

    for sensor in SENSOR_TYPES:
        sensor_dir = os.path.join(SRC_ROOT, sensor)
        if not os.path.isdir(sensor_dir):
            print(f"  [跳过] 子目录不存在：{sensor_dir}")
            continue

        bin_files = sorted(glob.glob(os.path.join(sensor_dir, "*.bin")))
        if not bin_files:
            print(f"  [跳过] {sensor} 目录下未找到 .bin 文件")
            continue

        for bin_path in bin_files:
            stem       = os.path.splitext(os.path.basename(bin_path))[0]
            label_path = os.path.join(sensor_dir, stem + ".label")

            seq_id = parse_filename(stem)
            if seq_id is None:
                print(f"  [WARN] 无法解析序列号，跳过：{stem}")
                n_unknown += 1
                continue

            if not os.path.exists(label_path):
                n_missing += 1
                continue

            seq_key = f"{seq_id}_{sensor}"
            seq_map.setdefault(seq_key, []).append((bin_path, label_path, sensor))
            n_total += 1

    print(f"\n  共找到 {len(seq_map)} 个序列，{n_total} 个有效帧"
          f"（缺 .label：{n_missing}，文件名异常：{n_unknown}）")

    print(f"\n  {'序列目录名':25s}  {'总帧数':>6}")
    print("  " + "-" * 35)
    for seq_key in sorted(seq_map.keys()):
        print(f"  {seq_key:25s}  {len(seq_map[seq_key]):>6}")

    return {k: seq_map[k] for k in sorted(seq_map.keys())}


# ══ 步骤 2：去重并写出 .bin / .label ══════════════════════════════════════════

def process_and_save(seq_map):
    total_ok = total_skip = 0

    for seq_key, triples in seq_map.items():
        vel_dir = os.path.join(DST_ROOT, "sequences", seq_key, "velodyne")
        lab_dir = os.path.join(DST_ROOT, "sequences", seq_key, "labels")
        os.makedirs(vel_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)

        print(f"\n[seq {seq_key}]  共 {len(triples)} 帧")

        for bin_path, label_path, sensor in triples:
            stem    = os.path.splitext(os.path.basename(bin_path))[0]
            out_bin = os.path.join(vel_dir, stem + ".bin")
            out_lbl = os.path.join(lab_dir, stem + ".label")

            try:
                raw_data   = read_bin(bin_path)                    # (N, 4) float32
                num_points = raw_data.shape[0]
                labels     = read_label(label_path, num_points)    # (N,)   int32（自动 dtype）

                # m1：去除 NaN 点，同步过滤标签
                nan_removed = 0
                if sensor == "m1":
                    raw_data, labels, nan_removed = remove_nan_points(raw_data, labels)
                    if nan_removed > 0:
                        print(f"  [NaN]  {stem}  "
                              f"去除 {nan_removed} 个 NaN 点，"
                              f"剩余 {raw_data.shape[0]} 点")

                # ouster：去除结构化雷达填充的 (0,0,0) 占位点
                zero_removed = 0
                if sensor == "ouster":
                    valid_mask   = ~((raw_data[:, 0] == 0) &
                                     (raw_data[:, 1] == 0) &
                                     (raw_data[:, 2] == 0))
                    zero_removed = int((~valid_mask).sum())
                    raw_data     = raw_data[valid_mask]
                    labels       = labels[valid_mask]
                    if zero_removed > 0:
                        print(f"  [ZERO] {stem}  "
                              f"去除 {zero_removed} 个零坐标点，"
                              f"剩余 {raw_data.shape[0]} 点")

                # 去重
                comb   = np.concatenate(
                    [raw_data, labels.reshape(-1, 1).astype(np.float32)], axis=1
                )
                unique = np.unique(comb, axis=0)
                u_raw  = unique[:, :4].reshape(-1).astype(np.float32)
                u_lab  = unique[:, 4].reshape(-1).astype(np.int32)

                u_raw.tofile(out_bin)
                u_lab.tofile(out_lbl)

                dust_cnt = int((u_lab == DUST_LABEL).sum())
                print(f"  [OK] {stem}  "
                      f"{num_points}→{unique.shape[0]} pts  "
                      f"(nan={nan_removed}, zero={zero_removed})  dust={dust_cnt}")
                total_ok += 1

            except Exception as e:
                print(f"  [ERR] {bin_path}: {e}")
                total_skip += 1

    print(f"\n完成：成功 {total_ok} 帧，失败 {total_skip} 帧")


# ══ 步骤 3：统计各序列 content ════════════════════════════════════════════════

def count_content(seq_map):
    global_counts = {}
    for seq_key in seq_map:
        lab_dir = os.path.join(DST_ROOT, "sequences", seq_key, "labels")
        for lbl_path in sorted(glob.glob(os.path.join(lab_dir, "*.label"))):
            lbls = np.fromfile(lbl_path, dtype=np.int32)
            for val in np.unique(lbls):
                n = int((lbls == val).sum())
                global_counts[int(val)] = global_counts.get(int(val), 0) + n
    return global_counts


# ══ 打印 yaml 参考片段 ════════════════════════════════════════════════════════

def print_yaml_hint(seq_map, global_counts):
    all_keys = sorted(seq_map.keys())

    print("\n" + "=" * 65)
    print("所有可用序列目录名（请按需分配到 yaml 的 train / valid / test）：")
    print("=" * 65)
    for k in all_keys:
        print(f"  - {k}")

    print("\n── 全局 content（所有序列汇总，写入 yaml content 字段）──")
    for cls_id, cnt in sorted(global_counts.items()):
        print(f"  {cls_id}: {cnt}")

    print("\n── yaml split 示例（序列号仅供参考，请自行调整）──")
    n       = len(all_keys)
    n_train = max(1, int(n * 0.70))
    n_val   = max(1, (n - n_train) // 2)
    print("split:")
    print("  train:")
    for k in all_keys[:n_train]:
        print(f"    - {k}")
    print("  valid:")
    for k in all_keys[n_train:n_train + n_val]:
        print(f"    - {k}")
    print("  test:")
    for k in all_keys[n_train + n_val:]:
        print(f"    - {k}")
    print("=" * 65)


# ══ 主流程 ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"输入根目录：{SRC_ROOT}")
    print(f"输出目录：  {DST_ROOT}")
    print(f"沙尘标签：  {DUST_LABEL}")
    print(f"传感器列表：{SENSOR_TYPES}\n")

    clean_output_dir()

    print("── 步骤 1：扫描文件并按序列号分组 ──")
    seq_map = collect_by_sequence()
    if not seq_map:
        raise RuntimeError("没有找到任何有效样本，请检查 SRC_ROOT 和 SENSOR_TYPES 配置")

    print("\n── 步骤 2：去重并写出 ──")
    process_and_save(seq_map)

    print("\n── 步骤 3：统计各类别点数 ──")
    global_counts = count_content(seq_map)

    print_yaml_hint(seq_map, global_counts)

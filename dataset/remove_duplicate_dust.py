"""
remove_duplicate_dust.py
针对 LIDARDUSTX 数据集的去重 + train/val/test 划分脚本

公平策略（适合写入学术论文）：
  1. 每个传感器独立按 70/15/15 随机划分，保证每种传感器三个集合中比例一致。
  2. 组装序列时，按各传感器在该集合中的实际帧数比例分配槽位
     （stratified interleave），使每个 sequence 内传感器构成比例完全一致。
     例：ls64:ls128:ouster = 500:300:100，则每个 sequence 内比例也是 5:3:1。
  3. 所有数据都被使用，不丢弃任何帧。

安全：脚本启动时自动清空输出目录，避免残留脏数据。

输入结构：
    LIDARDUSTX/
    ├── pcd/
    │   └── ls64/ ls128/ ly150/ ly300/ m1/ ouster/   *.pcd
    └── gt_txt/
        └── ls64/ ls128/ ly150/ ly300/ m1/ ouster/   *.txt

输出结构（与 WADS2 完全兼容）：
    LIDARDUSTX2/
    └── sequences/
        ├── 00/  velodyne/*.bin  labels/*.label
        ├── 01/  ...
        └── ...
"""

import os
import glob
import random
import shutil
import math
import numpy as np

# ── 配置区（按需修改）─────────────────────────────────────────────────────────
SRC_ROOT           = "/home/bbb/dataset/data/LIDARDUSTX"
DST_ROOT           = "/home/bbb/dataset/data/LIDARDUSTX2"

SENSOR_TYPES       = ["ls64", "ls128", "ly150", "ly300", "m1", "ouster"]

DUST_LABEL         = 110      # gt_txt 中沙尘点的类别 ID
RANDOM_SEED        = 42

TRAIN_RATIO        = 0.70
VAL_RATIO          = 0.15
TEST_RATIO         = 0.15

MAX_FRAMES_PER_SEQ = 300      # 每个 sequence 文件夹最多放多少帧
# ─────────────────────────────────────────────────────────────────────────────


# ══ 安全清空输出目录 ══════════════════════════════════════════════════════════

def clean_output_dir():
    if os.path.exists(DST_ROOT):
        print(f"[清理] 检测到旧输出目录 {DST_ROOT}，正在删除...")
        shutil.rmtree(DST_ROOT)
        print(f"[清理] 已删除旧目录。")
    os.makedirs(DST_ROOT, exist_ok=True)


# ══ I/O 工具 ══════════════════════════════════════════════════════════════════

def read_pcd(filepath):
    """读取 PCD，返回 (N,4) float32 [x,y,z,intensity]"""
    with open(filepath, 'rb') as f:
        fields, size, data_type = [], [], 'ascii'
        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('SIZE'):
                size = list(map(int, line.split()[1:]))
            elif line.startswith('DATA'):
                data_type = line.split()[1]
                break

        if data_type == 'ascii':
            data = np.loadtxt(f)
            if data.ndim == 1:
                data = data.reshape(1, -1)
        elif data_type == 'binary':
            dtype_map = {1: np.uint8, 2: np.uint16, 4: np.float32, 8: np.float64}
            dt = np.dtype([(fn, dtype_map.get(s, np.float32))
                           for fn, s in zip(fields, size)])
            raw  = np.frombuffer(f.read(), dtype=dt)
            data = np.column_stack([raw[fn].astype(np.float32) for fn in fields])
        elif data_type == 'binary_compressed':
            try:
                import open3d as o3d
                pc    = o3d.t.io.read_point_cloud(filepath)
                pts   = pc.point.positions.numpy()
                inten = (pc.point['intensity'].numpy().reshape(-1, 1)
                         if 'intensity' in pc.point
                         else np.zeros((len(pts), 1), dtype=np.float32))
                return np.hstack([pts, inten]).astype(np.float32)
            except ImportError:
                raise RuntimeError("binary_compressed 格式需要 open3d：pip install open3d")
        else:
            raise ValueError(f"不支持的 PCD data 类型: {data_type}")

    fl = [fn.lower() for fn in fields]
    xi = fl.index('x') if 'x' in fl else 0
    yi = fl.index('y') if 'y' in fl else 1
    zi = fl.index('z') if 'z' in fl else 2
    x = data[:, xi].reshape(-1, 1).astype(np.float32)
    y = data[:, yi].reshape(-1, 1).astype(np.float32)
    z = data[:, zi].reshape(-1, 1).astype(np.float32)
    if 'intensity' in fl:
        intensity = data[:, fl.index('intensity')].reshape(-1, 1).astype(np.float32)
    elif 'i' in fl:
        intensity = data[:, fl.index('i')].reshape(-1, 1).astype(np.float32)
    else:
        intensity = np.zeros_like(x)
    return np.hstack([x, y, z, intensity])


def remove_nan_and_remap(raw_data, labels):
    """
    去除 NaN 点，同时用旧→新索引映射表同步修正标签数组。

    原理：
        原始点云索引 0,1,2,3,4,5 中，假设第 1、3 号点是 NaN：
            旧索引:  0  1(NaN)  2  3(NaN)  4  5
            新索引:  0   ×      1    ×      2  3
        构建映射表 old2new = [0, -1, 1, -1, 2, 3]
        标签数组按旧索引取值后，用映射表写到新索引位置；
        落在 NaN 点上的标签直接丢弃（那个点本身已不存在）。

    参数：
        raw_data : (N, 4) float32，可能含 NaN
        labels   : (N,)   int32，与 raw_data 一一对应

    返回：
        clean_data   : (M, 4) float32，无 NaN，M <= N
        clean_labels : (M,)   int32，与 clean_data 一一对应
        n_removed    : 去除的 NaN 点数
    """
    # 任意维度为 NaN 的点都视为无效
    valid_mask = ~np.isnan(raw_data).any(axis=1)   # shape (N,)
    n_removed  = int((~valid_mask).sum())

    clean_data   = raw_data[valid_mask]
    clean_labels = labels[valid_mask]              # 直接用 bool 掩码同步过滤

    return clean_data, clean_labels, n_removed


def read_label_txt(filepath, num_points):
    """读取稀疏索引标注 txt，返回 (N,) int32"""
    labels = np.zeros(num_points, dtype=np.int32)
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            idx = int(parts[0].strip())
            cls = int(parts[1].strip())
            if idx < num_points:
                labels[idx] = cls
            else:
                print(f"  [WARN] 标签索引 {idx} 超出范围 {num_points}，跳过")
    return labels


# ══ 步骤 1：按传感器收集文件对 ════════════════════════════════════════════════

def collect_pairs_per_sensor():
    """返回 { sensor: [(pcd_path, txt_path), ...] }"""
    sensor_pairs = {}
    print(f"  {'传感器':8s}  {'有效样本':>8}  {'缺标注':>6}")
    print("  " + "-" * 30)
    for sensor in SENSOR_TYPES:
        pcd_dir = os.path.join(SRC_ROOT, "pcd",    sensor)
        txt_dir = os.path.join(SRC_ROOT, "gt_txt", sensor)
        if not os.path.isdir(pcd_dir):
            print(f"  {sensor:8s}  [目录不存在，跳过]")
            continue
        pairs, missing = [], 0
        for pcd_path in sorted(glob.glob(os.path.join(pcd_dir, "*.pcd"))):
            stem     = os.path.splitext(os.path.basename(pcd_path))[0]
            txt_path = os.path.join(txt_dir, stem + ".txt")
            if os.path.exists(txt_path):
                pairs.append((pcd_path, txt_path))
            else:
                missing += 1
        sensor_pairs[sensor] = pairs
        print(f"  {sensor:8s}  {len(pairs):>8}  {missing:>6}")
    return sensor_pairs


# ══ 步骤 2：每个传感器独立 70/15/15 划分 ══════════════════════════════════════

def split_per_sensor(sensor_pairs):
    """
    每个传感器独立随机划分。
    返回三个字典 { sensor: [(pcd, txt), ...] }，分别对应 train/val/test。
    """
    random.seed(RANDOM_SEED)
    train_by, val_by, test_by = {}, {}, {}

    print(f"\n  {'传感器':8s}  {'总数':>6}  {'train':>6}  {'val':>5}  {'test':>5}")
    print("  " + "-" * 40)
    for sensor, pairs in sensor_pairs.items():
        shuffled = pairs.copy()
        random.shuffle(shuffled)
        n       = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)
        train_by[sensor] = shuffled[:n_train]
        val_by[sensor]   = shuffled[n_train:n_train + n_val]
        test_by[sensor]  = shuffled[n_train + n_val:]
        print(f"  {sensor:8s}  {n:>6}  {n_train:>6}  "
              f"{len(val_by[sensor]):>5}  {len(test_by[sensor]):>5}")
    print("  " + "-" * 40)
    return train_by, val_by, test_by


# ══ 步骤 3：按比例分配槽位，保证每个 sequence 内传感器比例一致 ════════════════

def stratified_interleave(by_sensor):
    """
    按各传感器帧数比例分配 sequence 内槽位，再循环填充。

    原理：
      设各传感器帧数为 n_1, n_2, ..., n_k，总帧数 N = sum(n_i)。
      每个 sequence 的槽位数 = MAX_FRAMES_PER_SEQ。
      传感器 i 在每个 sequence 内分配 round(MAX_FRAMES_PER_SEQ * n_i/N) 个槽。
      因此每个 sequence 内传感器比例完全一致，反映真实数据分布。

    返回：[(pcd, txt, sensor), ...]  已按槽位顺序排列，直接切分成 sequences 即可。
    """
    # 只保留有数据的传感器
    active = {s: pairs for s, pairs in by_sensor.items() if len(pairs) > 0}
    if not active:
        return []

    total = sum(len(v) for v in active.values())

    # 计算每个 sequence 内各传感器的槽位数（至少 1 个槽）
    slots = {}
    for sensor, pairs in active.items():
        slots[sensor] = max(1, round(MAX_FRAMES_PER_SEQ * len(pairs) / total))

    # 打印槽位分配
    slot_total = sum(slots.values())
    print(f"  每个 sequence 内槽位分配（共 {slot_total} 槽）：")
    for sensor, s in sorted(slots.items()):
        pct = s / slot_total * 100
        print(f"    {sensor:8s}: {s:3d} 槽  ({pct:.1f}%)")

    # 用循环队列把所有帧按槽位顺序排列
    # 每轮：传感器 i 放 slots[i] 帧，然后轮到下一个传感器
    result = []
    queues = {s: list(pairs) for s, pairs in active.items()}
    sensors_order = list(active.keys())

    # 持续循环直到所有传感器的队列都为空
    while any(len(queues[s]) > 0 for s in sensors_order):
        for sensor in sensors_order:
            q = queues[sensor]
            n = slots[sensor]
            for _ in range(n):
                if not q:
                    break
                pcd_path, txt_path = q.pop(0)
                result.append((pcd_path, txt_path, sensor))

    return result


def assign_sequences(ordered_list, start_seq_id):
    """
    将有序帧列表按 MAX_FRAMES_PER_SEQ 切分成多个 sequence。
    因槽位已按比例排列，直接切分每个 sequence 内比例自动一致。
    """
    seq_map = {}
    seq_id  = start_seq_id
    for i in range(0, len(ordered_list), MAX_FRAMES_PER_SEQ):
        seq_map[seq_id] = ordered_list[i:i + MAX_FRAMES_PER_SEQ]
        seq_id += 1
    return seq_map, seq_id


# ══ 步骤 4：去重并写出 .bin / .label ══════════════════════════════════════════

def process_and_save(all_seq_map):
    total_ok = total_skip = 0

    for seq_id, triples in sorted(all_seq_map.items()):
        seq_str = str(seq_id).zfill(2)
        vel_dir = os.path.join(DST_ROOT, "sequences", seq_str, "velodyne")
        lab_dir = os.path.join(DST_ROOT, "sequences", seq_str, "labels")
        os.makedirs(vel_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)

        # 打印该序列传感器分布摘要
        sensor_cnt = {}
        for _, _, sensor in triples:
            sensor_cnt[sensor] = sensor_cnt.get(sensor, 0) + 1
        dist_str = "  ".join(f"{s}:{c}" for s, c in sorted(sensor_cnt.items()))
        print(f"\n[seq{seq_str}]  共 {len(triples)} 帧  分布: {dist_str}")

        for pcd_path, txt_path, sensor in triples:
            stem        = os.path.splitext(os.path.basename(pcd_path))[0]
            unique_stem = f"{sensor}_{stem}"
            bin_path    = os.path.join(vel_dir, unique_stem + ".bin")
            lbl_path    = os.path.join(lab_dir, unique_stem + ".label")

            try:
                raw_data   = read_pcd(pcd_path)
                num_points = raw_data.shape[0]
                labels     = read_label_txt(txt_path, num_points)

                # M1：去除 NaN 点，同步修正标签索引
                nan_removed = 0
                if sensor == "m1":
                    raw_data, labels, nan_removed = remove_nan_and_remap(raw_data, labels)
                    if nan_removed > 0:
                        print(f"  [NaN]  [{sensor}] {stem}  去除 {nan_removed} 个 NaN 点，"
                              f"剩余 {raw_data.shape[0]} 点")

                # ouster：去除零坐标占位点 (0,0,0)
                # ouster 是结构化雷达，对无效像素统一填充 (0,0,0)
                # 这些点的 intensity 并不全为 0，np.unique 不会自动合并，
                # 必须在去重前显式过滤，否则会当作有效点保留
                zero_removed = 0
                if sensor == "ouster":
                    valid_mask   = ~((raw_data[:, 0] == 0) &
                                     (raw_data[:, 1] == 0) &
                                     (raw_data[:, 2] == 0))
                    zero_removed = int((~valid_mask).sum())
                    raw_data     = raw_data[valid_mask]
                    labels       = labels[valid_mask]
                    if zero_removed > 0:
                        print(f"  [ZERO] [{sensor}] {stem}  去除 {zero_removed} 个零坐标点，"
                              f"剩余 {raw_data.shape[0]} 点")

                # 去重
                comb   = np.concatenate(
                    [raw_data, labels.reshape(-1, 1).astype(np.float32)], axis=1
                )
                unique = np.unique(comb, axis=0)
                u_raw  = unique[:, :4].reshape(-1).astype(np.float32)
                u_lab  = unique[:, 4].reshape(-1).astype(np.int32)

                u_raw.tofile(bin_path)
                u_lab.tofile(lbl_path)

                dust_cnt = (u_lab == DUST_LABEL).sum()
                print(f"  [OK] [{sensor}] {stem}  "
                      f"{num_points}→{unique.shape[0]} pts  "
                      f"(nan={nan_removed}, zero={zero_removed})  dust={dust_cnt}")
                total_ok += 1
            except Exception as e:
                print(f"  [ERR] {pcd_path}: {e}")
                total_skip += 1

    print(f"\n完成：成功 {total_ok} 帧，失败 {total_skip} 帧")


# ══ 步骤 5：统计训练集 content ════════════════════════════════════════════════

def count_content(train_seq_map):
    counts = {}
    for seq_id in train_seq_map:
        seq_str = str(seq_id).zfill(2)
        lab_dir = os.path.join(DST_ROOT, "sequences", seq_str, "labels")
        for lbl_path in glob.glob(os.path.join(lab_dir, "*.label")):
            labels = np.fromfile(lbl_path, dtype=np.int32)
            for val in np.unique(labels):
                counts[int(val)] = counts.get(int(val), 0) + int((labels == val).sum())
    return counts


# ══ 打印 yaml 片段 ════════════════════════════════════════════════════════════

def print_yaml_snippet(train_seqs, val_seqs, test_seqs, content_counts):
    print("\n" + "=" * 60)
    print("请将以下内容写入你的 yaml 配置文件：")
    print("=" * 60)
    print("\ncontent:")
    for cls_id, cnt in sorted(content_counts.items()):
        print(f"  {cls_id}: {cnt}")
    print("\nsplit:")
    print("  train:")
    for s in train_seqs:
        print(f"    - {s}")
    print("  valid:")
    for s in val_seqs:
        print(f"    - {s}")
    print("  test:")
    for s in test_seqs:
        print(f"    - {s}")
    print("=" * 60)


# ══ 主流程 ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"源数据集：{SRC_ROOT}")
    print(f"输出目录：{DST_ROOT}")
    print(f"划分比例：train={TRAIN_RATIO}  val={VAL_RATIO}  test={TEST_RATIO}"
          f"  （每个传感器独立划分）")
    print(f"每序列最大帧数：{MAX_FRAMES_PER_SEQ}\n")

    # 0. 安全清空旧输出
    clean_output_dir()

    # 1. 按传感器收集文件对
    print("── 步骤 1：收集文件 ──")
    sensor_pairs = collect_pairs_per_sensor()
    if sum(len(v) for v in sensor_pairs.values()) == 0:
        raise RuntimeError("没有找到任何有效样本，请检查 SRC_ROOT 和 SENSOR_TYPES")

    # 2. 每个传感器独立 70/15/15 划分
    print("\n── 步骤 2：独立划分各传感器 ──")
    train_by, val_by, test_by = split_per_sensor(sensor_pairs)

    # 3. 按比例槽位交替排列
    print("\n── 步骤 3：按比例分配序列槽位 ──")
    print("  [train]")
    train_ordered = stratified_interleave(train_by)
    print("  [val]")
    val_ordered   = stratified_interleave(val_by)
    print("  [test]")
    test_ordered  = stratified_interleave(test_by)

    # 4. 分配序列号
    train_seq_map, next_id = assign_sequences(train_ordered, start_seq_id=0)
    val_seq_map,   next_id = assign_sequences(val_ordered,   start_seq_id=next_id)
    test_seq_map,  next_id = assign_sequences(test_ordered,  start_seq_id=next_id)
    all_seq_map = {**train_seq_map, **val_seq_map, **test_seq_map}

    print(f"\n序列号分配：")
    print(f"  train → seq {sorted(train_seq_map.keys())}")
    print(f"  val   → seq {sorted(val_seq_map.keys())}")
    print(f"  test  → seq {sorted(test_seq_map.keys())}")

    # 5. 去重并写出
    print("\n── 步骤 4：去重并写出 ──")
    process_and_save(all_seq_map)

    # 6. 统计训练集 content
    print("\n── 步骤 5：统计训练集 content ──")
    content_counts = count_content(train_seq_map)

    # 7. 打印 yaml 片段
    print_yaml_snippet(
        sorted(train_seq_map.keys()),
        sorted(val_seq_map.keys()),
        sorted(test_seq_map.keys()),
        content_counts
    )

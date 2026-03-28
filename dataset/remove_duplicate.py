import numpy as np
import os

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset._point_cloud_dataset import get_files

# ── PCD 读取函数 ────────────────────────────────────────────────────────────
def read_pcd(filepath):
    """
    读取 PCD 文件，返回 (N, 4) 的 numpy 数组 [x, y, z, intensity]
    支持 ASCII 和 binary 格式，intensity 字段不存在时补 0
    """
    with open(filepath, 'rb') as f:
        # 解析 header
        fields = []
        size = []
        count = []
        data_type = 'ascii'
        num_points = 0

        while True:
            line = f.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('FIELDS'):
                fields = line.split()[1:]
            elif line.startswith('SIZE'):
                size = list(map(int, line.split()[1:]))
            elif line.startswith('COUNT'):
                count = list(map(int, line.split()[1:]))
            elif line.startswith('DATA'):
                data_type = line.split()[1]
                break
            elif line.startswith('POINTS'):
                num_points = int(line.split()[1])

        # 读取数据
        if data_type == 'ascii':
            data = np.loadtxt(f)
            if data.ndim == 1:
                data = data.reshape(1, -1)
        elif data_type == 'binary':
            # 构建 dtype
            dtype_map = {1: np.uint8, 2: np.uint16, 4: np.float32, 8: np.float64}
            dt = np.dtype([(f, dtype_map.get(s, np.float32))
                           for f, s in zip(fields, size)])
            raw = np.frombuffer(f.read(), dtype=dt)
            data = np.column_stack([raw[field].astype(np.float32)
                                    for field in fields])
        elif data_type == 'binary_compressed':
            # 需要 lzf 解压，使用 open3d 作为 fallback
            try:
                import open3d as o3d
                pcd_o3d = o3d.t.io.read_point_cloud(filepath)
                pts = pcd_o3d.point.positions.numpy()
                if 'intensity' in pcd_o3d.point:
                    inten = pcd_o3d.point['intensity'].numpy().reshape(-1, 1)
                else:
                    inten = np.zeros((len(pts), 1), dtype=np.float32)
                return np.hstack([pts, inten]).astype(np.float32)
            except ImportError:
                raise RuntimeError(
                    "binary_compressed PCD 需要 open3d，请执行: pip install open3d")
        else:
            raise ValueError(f"不支持的 PCD data 类型: {data_type}")

    # 提取 x y z
    field_lower = [f.lower() for f in fields]
    xi = field_lower.index('x') if 'x' in field_lower else 0
    yi = field_lower.index('y') if 'y' in field_lower else 1
    zi = field_lower.index('z') if 'z' in field_lower else 2

    x = data[:, xi].reshape(-1, 1).astype(np.float32)
    y = data[:, yi].reshape(-1, 1).astype(np.float32)
    z = data[:, zi].reshape(-1, 1).astype(np.float32)

    # 提取 intensity（找 intensity 或 i 字段）
    if 'intensity' in field_lower:
        ii = field_lower.index('intensity')
        intensity = data[:, ii].reshape(-1, 1).astype(np.float32)
    elif 'i' in field_lower:
        ii = field_lower.index('i')
        intensity = data[:, ii].reshape(-1, 1).astype(np.float32)
    else:
        intensity = np.zeros_like(x)

    return np.hstack([x, y, z, intensity])  # (N, 4)


def read_label_txt(filepath, num_points):
    """
    读取稀疏索引格式的 txt 标签文件，返回 (N, 1) 的 int32 数组

    文件格式（每行）：点云索引, 类别ID
        194, 110
        200, 110
        ...
    未出现在文件中的点默认标签为 0（正常点）
    出现在文件中的点标签为对应类别 ID（110 = 噪声点）

    参数:
        filepath:   txt 文件路径
        num_points: 对应点云的总点数（从 pcd 读取）
    """
    labels = np.zeros(num_points, dtype=np.int32)  # 默认全部为 0（正常点）

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
                print(f"[WARN] 标签索引 {idx} 超出点云范围 {num_points}，跳过")

    return labels.reshape(-1, 1)


# ── 主流程 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    src_root = "/home/bbb/dataset/data/WADS/sequences"
    dst_root = "/home/bbb/dataset/data/WADS2/sequences"

    split = [15, 18, 36, 12, 17, 22, 26, 28, 34,
             11, 16, 13, 23, 14, 20, 24, 30, 35, 37, 76]

    # 收集所有 pcd 文件路径
    im_idx = list()
    for i_folder in split:
        velodyne_dir = '/'.join([src_root, str(i_folder).zfill(2), 'velodyne'])
        im_idx += get_files(velodyne_dir, 'pcd')   # ← 改为 pcd

    print(f"共找到 {len(im_idx)} 个 pcd 文件")

    for im in im_idx:
        # ── 读取点云（pcd）──────────────────────
        raw_data = read_pcd(im)          # (N, 4)  float32

        # ── 读取标签（txt）──────────────────────
        label_path = im.replace('velodyne', 'labels')
        # 去掉 .pcd 后缀，加 .txt
        label_path = os.path.splitext(label_path)[0] + '.txt'
        num_points = raw_data.shape[0]
        annotated_data = read_label_txt(label_path, num_points)  # (N, 1) int32

        # ── 去重 ────────────────────────────────
        comb = np.concatenate((raw_data, annotated_data), axis=1)
        unique = np.unique(comb, axis=0)
        u_raw = unique[:, 0:4].reshape(-1).astype(np.float32)
        u_lab = unique[:, 4].reshape(-1).astype(np.int32)

        # ── 输出路径（写 .bin + .label 到 WADS2）──
        # 把 pcd 文件路径的 WADS → WADS2，扩展名 .pcd → .bin
        u_im_file = im.replace("WADS", "WADS2")
        u_im_file = os.path.splitext(u_im_file)[0] + '.bin'

        u_lab_file = u_im_file.replace('velodyne', 'labels')
        u_lab_file = os.path.splitext(u_lab_file)[0] + '.label'

        os.makedirs(os.path.dirname(u_im_file), exist_ok=True)
        os.makedirs(os.path.dirname(u_lab_file), exist_ok=True)

        # ── 写出二进制格式（后续训练代码直接读）──
        u_raw.tofile(u_im_file)
        u_lab.tofile(u_lab_file)

        print(f"[OK] {os.path.basename(im)}  {unique.shape[0]} pts → {u_im_file}")

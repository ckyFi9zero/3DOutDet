import torch
from torch.utils import data
import yaml
import os
import glob
import numpy as np
import pickle
import cupy as cp
from cuml.neighbors import NearestNeighbors
try:
    from cuml.common.device_selection import set_global_device_type
except ImportError:
    def set_global_device_type(*args, **kwargs):
        return None


set_global_device_type('gpu')


def get_files(folder, ext):
    files = glob.glob(os.path.join(folder, f"*.{ext}"))
    return files


class  WadsPointCloudDataset(data.Dataset):
    def __init__(self, device, data_path, imageset='train', label_conf='wads.yaml', k=121, leaf_size=100, mean=[0.3420934,  -0.01516175 ,-0.5889243 ,  9.875928  ], std=[25.845459,  18.93466,    1.5863657, 14.734034 ],
                 shuffle_indices=False, save_ind=True, recalculate=False, desnow_root=None, pred_folder=None,
                                         snow_label=None):
        self.device = device
        self.recalculate = recalculate
        self.k = k
        self.leaf_size = leaf_size
        self.save_ind = save_ind
        self.mean = np.array(mean)
        self.std = np.array(std)
        with open(label_conf, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        self.shuffle_indices = shuffle_indices
        self.desnow_root = desnow_root
        self.pred_folder = pred_folder
        self.snow_label = snow_label
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        elif imageset == 'all':
            split = semkittiyaml['split']['train'] + semkittiyaml['split']['valid'] + semkittiyaml['split']['test']
        elif imageset == 'bug':
            split = ["05"]
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.pred_idx = list()
        for i_folder in split:
            self.im_idx += get_files('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), 'bin')
            if desnow_root is not None:
                assert os.path.exists(desnow_root)
                self.pred_idx += get_files('/'.join([desnow_root, str(i_folder).zfill(2), pred_folder]), 'label')

        self.im_idx.sort()
        if desnow_root is not None:
            self.pred_idx.sort()


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                     dtype=np.int32).reshape((-1, 1))
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
        if self.imageset == 'train':
            if np.random.random() > 0.5:
                rotate_rad = np.deg2rad(np.random.random()*360)
                cos, sine = np.cos(rotate_rad), np.sin(rotate_rad)
                rot_mat = np.matrix([[cos, sine], [-sine, cos]])
                # rotate x and y
                data[:, :2] = np.dot(data[:, :2], rot_mat)
            # if np.random.uniform(high=1.0, low=0.0) >= 0.5:
            #     data[:, 0] *= -1
            # if np.random.uniform(high=1.0, low=0.0) >= 0.5:
            #     data[:, 1] *= -1
            #
            # if np.random.uniform(high=1.0, low=0.0) >= 0.5:
            #     data[:, 2] *= -1
        # data = raw_data

        if self.desnow_root is not None:
            if self.pred_idx[index].endswith('.pred'):
                preds = np.fromfile(self.pred_idx[index], dtype=np.int64)
            else:
                preds = np.fromfile(self.pred_idx[index], dtype=np.int32)
            preds = preds.reshape(-1)
            snow_indices = np.where(preds == self.snow_label)[0]

            data = np.delete(data, obj=snow_indices.tolist(), axis=0)
            annotated_data = np.delete(annotated_data, obj=snow_indices.tolist(), axis=0)
        kd_path = self.im_idx[index].replace('velodyne', 'knn')[:-3] + 'pkl'
        # err = True
        if os.path.exists(kd_path) and not self.recalculate:
            with open(kd_path, 'rb') as f:
                try:
                    ind = pickle.load(f)
                    dist = pickle.load(f)
                    # if ind.shape[1] > self.k:
                    #     ind = ind[:, :self.k]
                    #     dist = dist[:, :self.k]
                    err = False
                except EOFError:
                    err = True
        else:
            err = True
        if err or self.recalculate:
            p1 = torch.from_numpy(data[:, :3]).to(self.device)
            p1 = cp.asarray(p1)
            # metric: string(default='euclidean').
            # Supported
            # distances
            # are['l1, '
            # cityblock
            # ',
            # 'taxicab', 'manhattan', 'euclidean', 'l2', 'braycurtis', 'canberra',
            # 'minkowski', 'chebyshev', 'jensenshannon', 'cosine', 'correlation']
            self.nn = NearestNeighbors()
            while True:

                try:
                    self.nn.fit(p1)
                    break
                except Exception:
                    print("caught it")

            dist, ind = self.nn.kneighbors(p1, self.k)
            # ['euclidean', 'l2', 'minkowski', 'p', 'manhattan', 'cityblock', 'l1', 'chebyshev', 'infinity']
            # tree = KDTree(data[:, :3], leaf_size=self.leaf_size, metric='cityblock')


            # ind, dist = tree.query_radius(data[:,:3], r=0.5, return_distance=True)
            # process radius and ind for dist query
            # dist = uneven_stack(dist, limit=self.k)
            # ind = uneven_stack(ind, limit=self.k)
            # dist, ind = tree.query(data[:, :3], k=self.k)
            ind = cp.asnumpy(ind)
            dist = cp.asnumpy(dist)
            ind = ind.astype(np.int64)
            # dist = dist.reshape(data.shape[0], -1)
            if self.save_ind:
                parent = os.path.dirname(kd_path)
                os.makedirs(parent, exist_ok=True)
                with open(kd_path, 'wb') as f:
                    pickle.dump(ind, f)
                    pickle.dump(dist, f)
        dist = dist + 1.0
        # normalize the distance
        # d_mean = np.mean(dist, axis=1, keepdims=True)
        # d_std = np.std(dist, axis=1, keepdims=True)
        # dist = (dist - d_mean) / d_std
        if self.shuffle_indices:
            s_ind = np.random.rand(*ind.shape).argsort(axis=1)
            ind = np.take_along_axis(ind, s_ind, axis=1)
            dist = np.take_along_axis(dist, s_ind, axis=1)


        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data).reshape(-1)
        data = (data - self.mean) / self.std

        if self.imageset == 'train':
            if np.random.random() > 0.5:
                data[:, :3] += np.random.normal(size=(data.shape[0], 3), loc=0, scale=0.1)
        out_dict = {'data': data.astype(np.float32), 'dist': dist.astype(np.float32), 'ind': ind, 'label': annotated_data.astype(np.uint8)}
        return out_dict


class PointCloudDataset(data.Dataset):
    def __init__(self, data_path, imageset='train', label_conf='wads.yaml'):
        self.mean = np.array([0.43,0.29,-0.67,10.8])
        self.std = np.array([1.17,1.40,0.05,0.97])
        with open(label_conf, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        elif imageset == 'all':
            split = semkittiyaml['split']['train'] + semkittiyaml['split']['valid'] + semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.pred_idx = list()
        for i_folder in split:
            self.im_idx += get_files('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), 'bin')
        self.im_idx.sort()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        data = (raw_data - self.mean) / self.std
        annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                     dtype=np.int32).reshape((-1, 1))
        annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary

        annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data).reshape(-1)

        out_dict = {'data': data.astype(np.float32), 'label': annotated_data.astype(np.uint8)}
        return out_dict


def get_files(folder, ext):
    files = glob.glob(os.path.join(folder, f"*.{ext}"))
    return files


class DustPointCloudDataset(data.Dataset):
    """
    LIDARDUSTX2 数据集加载器。

    与 WadsPointCloudDataset 的主要区别：
      1. mean / std 使用由 compute_mean_std.py 计算的沙尘数据集统计值
      2. intensity 在归一化前 clip 到 [0, 255]，消除跨传感器量程差异的影响
      3. 序列目录名为字符串格式（如 "0111_ls64"），不再做 zfill(2) 处理
      4. 去尘预测根目录参数改名为 dedust_root / dust_label，语义更清晰
      5. 移除已废弃的 np.matrix，改用 np.array
      6. 清理所有失效的注释代码
    """

    def __init__(
        self,
        device,
        data_path,
        imageset='train',
        label_conf='binary_dedust.yaml',
        k=121,
        leaf_size=100,
        # ── 由 compute_mean_std.py 计算得到，仅需修改这里 ──────────────────
        mean=[14.9649432, -0.0154078, -0.1577283, 41.5470127],
        std= [21.6694211,  15.8650666,  3.3049342, 52.9999838],
        # ──────────────────────────────────────────────────────────────────
        intensity_clip=255.0,       # intensity 归一化前的截断上限，None 表示不截断
        shuffle_indices=False,
        save_ind=True,
        recalculate=False,
        dedust_root=None,           # 预测结果根目录（推理后去尘时使用）
        pred_folder=None,           # 预测 label 子文件夹名
        dust_label=None,            # 预测文件中沙尘点的类别 ID
    ):
        self.device          = device
        self.recalculate     = recalculate
        self.k               = k
        self.leaf_size       = leaf_size
        self.save_ind        = save_ind
        self.mean            = np.array(mean, dtype=np.float32)
        self.std             = np.array(std,  dtype=np.float32)
        self.intensity_clip  = intensity_clip
        self.imageset        = imageset
        self.shuffle_indices = shuffle_indices
        self.dedust_root     = dedust_root
        self.pred_folder     = pred_folder
        self.dust_label      = dust_label

        # ── 读取 yaml 配置 ──────────────────────────────────────────────────
        with open(label_conf, 'r') as f:
            cfg = yaml.safe_load(f)
        self.learning_map = cfg['learning_map']

        # ── 按 imageset 选择序列列表 ────────────────────────────────────────
        if imageset == 'train':
            split = cfg['split']['train']
        elif imageset == 'val':
            split = cfg['split']['valid']
        elif imageset == 'test':
            split = cfg['split']['test']
        elif imageset == 'all':
            split = (cfg['split']['train'] +
                     cfg['split']['valid'] +
                     cfg['split']['test'])
        else:
            raise ValueError(f"imageset 必须是 train / val / test / all，got: {imageset}")

        # ── 收集所有 .bin 文件路径 ──────────────────────────────────────────
        # 序列目录名为字符串（如 "0111_ls64"），直接拼接，不做 zfill
        self.im_idx  = []
        self.pred_idx = []

        for seq in split:
            seq_str  = str(seq)
            vel_dir  = os.path.join(data_path, seq_str, 'velodyne')
            self.im_idx += get_files(vel_dir, 'bin')

            if dedust_root is not None:
                assert os.path.exists(dedust_root), \
                    f"dedust_root 不存在：{dedust_root}"
                pred_dir = os.path.join(dedust_root, seq_str, pred_folder)
                self.pred_idx += get_files(pred_dir, 'label')

        self.im_idx.sort()
        if dedust_root is not None:
            self.pred_idx.sort()

        print(f"[DustPointCloudDataset] imageset={imageset}  "
              f"sequences={len(split)}  frames={len(self.im_idx)}")

    # ── 数据集长度 ────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.im_idx)

    # ── 单帧读取 ──────────────────────────────────────────────────────────────

    def __getitem__(self, index):
        # ① 读取点云 (N, 4)  [x, y, z, intensity]
        data = np.fromfile(
            self.im_idx[index], dtype=np.float32
        ).reshape(-1, 4)

        # ② 读取标签 (N,)，只保留低 16 位语义标签
        label_path = (
            self.im_idx[index]
            .replace('velodyne', 'labels')[:-3] + 'label'
        )
        annotated_data = (
            np.fromfile(label_path, dtype=np.int32)
            .reshape(-1, 1) & 0xFFFF
        )

        # ③ 训练期间随机旋转增强（绕 z 轴 360°）
        if self.imageset == 'train':
            if np.random.random() > 0.5:
                angle    = np.deg2rad(np.random.random() * 360)
                cos, sin = np.cos(angle), np.sin(angle)
                rot_mat  = np.array([[cos, sin],
                                     [-sin, cos]], dtype=np.float32)
                data[:, :2] = data[:, :2] @ rot_mat.T

        # ④ 用预测结果去除已检出的沙尘点（推理后评估管线用）
        if self.dedust_root is not None:
            pred_path = self.pred_idx[index]
            preds = np.fromfile(
                pred_path,
                dtype=np.int64 if pred_path.endswith('.pred') else np.int32
            ).reshape(-1)
            dust_idx = np.where(preds == self.dust_label)[0]
            data           = np.delete(data,           dust_idx, axis=0)
            annotated_data = np.delete(annotated_data, dust_idx, axis=0)

        # ⑤ kNN 索引：读缓存 or 重新计算
        kd_path = (
            self.im_idx[index]
            .replace('velodyne', 'knn')[:-3] + 'pkl'
        )
        ind, dist, err = None, None, True

        if os.path.exists(kd_path) and not self.recalculate:
            with open(kd_path, 'rb') as f:
                try:
                    ind  = pickle.load(f)
                    dist = pickle.load(f)
                    err  = False
                except EOFError:
                    err = True

        if err:
            p1 = cp.asarray(
                torch.from_numpy(data[:, :3]).to(self.device)
            )
            nn = NearestNeighbors()
            while True:
                try:
                    nn.fit(p1)
                    break
                except Exception:
                    print("[kNN] cuML fit 失败，重试…")

            dist, ind = nn.kneighbors(p1, self.k)
            ind  = cp.asnumpy(ind).astype(np.int64)
            dist = cp.asnumpy(dist)

            if self.save_ind:
                os.makedirs(os.path.dirname(kd_path), exist_ok=True)
                with open(kd_path, 'wb') as f:
                    pickle.dump(ind,  f)
                    pickle.dump(dist, f)

        # ⑥ 距离平移（原始逻辑保持不变）
        dist = dist + 1.0

        # ⑦ 可选：随机打乱邻居顺序
        if self.shuffle_indices:
            s_ind = np.random.rand(*ind.shape).argsort(axis=1)
            ind   = np.take_along_axis(ind,  s_ind, axis=1)
            dist  = np.take_along_axis(dist, s_ind, axis=1)

        # ⑧ intensity clip：压制跨传感器量程差异带来的离群值
        if self.intensity_clip is not None:
            data[:, 3] = np.clip(data[:, 3], 0.0, self.intensity_clip)

        # ⑨ 标签映射
        annotated_data = np.vectorize(
            self.learning_map.__getitem__
        )(annotated_data).reshape(-1)

        # ⑩ 归一化
        data = (data - self.mean) / self.std

        # ⑪ 训练期间随机高斯噪声增强（xyz 通道）
        if self.imageset == 'train':
            if np.random.random() > 0.5:
                data[:, :3] += np.random.normal(
                    size=(data.shape[0], 3), loc=0.0, scale=0.1
                ).astype(np.float32)

        return {
            'data':  data.astype(np.float32),
            'dist':  dist.astype(np.float32),
            'ind':   ind,
            'label': annotated_data.astype(np.uint8),
        }

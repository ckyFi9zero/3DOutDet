#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import sys
import numpy as np
import warnings
import torch

from tqdm import tqdm
from dataset.utils.collate import collate_fn_cp
from deterministic import configure_randomness
from dataset import WadsPointCloudDataset   # 数据集结构兼容，直接复用

warnings.filterwarnings("ignore")


def main(args):
    device   = torch.device(args.device)
    tree_k   = int(np.round(args.K * args.K))   # K=3 → tree_k=9

    # imageset='all' 会读取 yaml 里 train+valid+test 全部序列
    # 对所有帧预计算 kNN 并缓存到 knn/ 子目录
    dataset = WadsPointCloudDataset(
        device,
        args.data_dir + '/sequences/',
        imageset='all',
        label_conf=args.label_config,
        k=tree_k,
        shuffle_indices=False,
        save_ind=True,
        recalculate=True,       # 强制重新计算，不读旧缓存
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_cp,
    )

    print(f"共 {len(dataset)} 帧，开始预计算 kNN（K={args.K}, tree_k={tree_k}）...")
    pbar = tqdm(total=len(dataset))
    for _ in loader:
        pbar.update()
    pbar.close()
    print("kNN 预计算完成，缓存已写入各序列的 knn/ 子目录。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为 LIDARDUSTX2 预计算 kNN 索引')

    parser.add_argument('-d', '--data_dir',
                        default='/home/bbb/dataset/data/LIDARDUSTX2',
                        help='LIDARDUSTX2 根目录（包含 sequences/ 子目录）')
    parser.add_argument('--label_config',
                        type=str,
                        default='binary_dedust.yaml',
                        help='yaml 配置文件路径')
    parser.add_argument('--K',
                        type=int,
                        default=3,
                        help='NH Convolution 的核大小，tree_k = K*K（默认 3 → 9 邻居）')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0')

    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)

    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)

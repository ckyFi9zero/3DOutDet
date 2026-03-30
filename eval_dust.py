#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import sys
import numpy as np
import torch
import yaml
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix

from deterministic import configure_randomness
from modules import OutDet
from dataset.utils.collate import collate_fn_cp, collate_fn_cp_inference
from dataset import WadsPointCloudDataset
import warnings

warnings.filterwarnings("ignore")


def get_seq_name_from_path(path):
    tmps  = path.split(os.path.sep)
    seq   = tmps[-3]
    name  = os.path.splitext(tmps[-1])[0]
    return seq, name


def main(args):
    data_path       = args.data_dir
    model_save_path = args.model_save_path
    device          = torch.device(args.device)
    dilate          = 1

    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)

    class_strings   = config["labels"]
    class_inv_remap = config["learning_map_inv"]
    num_classes     = len(class_inv_remap)

    # 预测类别索引 → 原始标签值 的查找表（用于保存预测文件）
    max_key      = max(class_inv_remap.keys())
    look_up_table = np.zeros((max_key + 1), dtype=np.int32)
    for k, v in class_inv_remap.items():
        look_up_table[k] = v

    ordered_class_names = [class_strings[class_inv_remap[i]] for i in range(num_classes)]

    # 模型
    model = OutDet(num_classes=num_classes, kernel_size=args.K, depth=args.depth, dilate=dilate)
    model = model.to(device)
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"已加载模型：{model_save_path}")
    else:
        raise FileNotFoundError(f"模型文件不存在：{model_save_path}")

    # 数据集（测试集，recalculate=True 实时计算 kNN，save_ind=False 不保存缓存）
    tree_k = int(np.round(args.K * args.K))
    test_dataset = WadsPointCloudDataset(
        device,
        data_path + '/sequences/',
        imageset='test',
        label_conf=args.label_config,
        k=tree_k,
        recalculate=True,
        save_ind=False,
    )

    collate_fn = collate_fn_cp if test_dataset.save_ind else collate_fn_cp_inference
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print('*' * 80)
    # 传感器过滤：文件名格式为 {sensor}_{stem}.bin，前缀即传感器名
    filter_sensor = args.sensor.lower() if args.sensor else None
    if filter_sensor:
        valid_sensors = ["ls64", "ls128", "ly150", "ly300", "m1", "ouster"]
        if filter_sensor not in valid_sensors:
            raise ValueError(f"--sensor 必须是以下之一: {valid_sensors}")
        sensor_frames = [p for p in test_dataset.im_idx
                         if os.path.basename(p).startswith(filter_sensor + '_')]
        print(f'传感器过滤：仅评估 [{filter_sensor}]，'
              f'共 {len(sensor_frames)}/{len(test_dataset)} 帧')
    else:
        print(f'测试集共 {len(test_dataset)} 帧，开始评估（全部传感器）...')
    print('*' * 80)

    pbar       = tqdm(total=len(test_loader))
    model.eval()
    pcms       = np.zeros(shape=(num_classes, 2, 2), dtype=np.int64)
    skip_count = 0
    eval_count = 0

    with torch.no_grad():
        for i_iter, batch in enumerate(test_loader):
            # 当指定了 --sensor 时，跳过不匹配的帧
            fname = os.path.basename(test_dataset.im_idx[i_iter])
            if filter_sensor and not fname.startswith(filter_sensor + '_'):
                pbar.update(1)
                skip_count += 1
                continue

            data  = batch['data'][0].to(device)
            ind   = batch['ind'][0]
            dist  = batch['dist'][0].to(device)
            label = batch['label'][0].long().to(device)

            logit          = model(data, dist, ind)
            predict_labels = torch.argmax(logit, dim=1)

            pcm = multilabel_confusion_matrix(
                y_true=label.cpu().numpy(),
                y_pred=predict_labels.cpu().numpy(),
                labels=[i for i in range(num_classes)]
            )
            pcms += pcm
            eval_count += 1

            # 保存预测标签文件（--save_pred 时启用）
            if args.save_pred:
                pred_np    = predict_labels.cpu().numpy().reshape(-1)
                inv_labels = look_up_table[pred_np].astype(np.int32)
                path_seq, name = get_seq_name_from_path(test_dataset.im_idx[i_iter])
                out_path = os.path.join(
                    args.test_output_path, "sequences",
                    path_seq, "predictions", name + ".label"
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                inv_labels.tofile(out_path)

            pbar.update(1)
    pbar.close()
    print(f'实际评估：{eval_count} 帧，跳过：{skip_count} 帧')

    # ── 打印评估结果 ────────────────────────────────────────────────────────────
    print('*' * 80)
    print('Evaluation using multilabel confusion matrix')
    print('*' * 80)

    IOUs = []
    for i in range(num_classes):
        iou = evaluate_cm(pcms[i], ordered_class_names[i])
        print(pcms[i])
        IOUs.append(iou)

    class_jaccard = torch.tensor(np.array(IOUs))
    m_jaccard     = class_jaccard.mean().item()

    # 论文格式输出（与原版一致，可直接贴入 LaTeX 表格）
    print('\n── 论文格式输出 ──')
    for i, jacc in enumerate(class_jaccard):
        sys.stdout.write('{jacc:.2f} &'.format(jacc=jacc.item() * 100))
    sys.stdout.write('{jacc:.2f}'.format(jacc=m_jaccard * 100))
    sys.stdout.write('\n')
    for i in range(1, len(class_jaccard)):
        sys.stdout.write('\\bfseries{{ {name} }} &'.format(name=ordered_class_names[i]))
    sys.stdout.write('\n')
    sys.stdout.flush()


def evaluate_cm(cm, class_name):
    """计算 Precision / Recall / IOU / F1，加 1e-9 防止除零"""
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    iou1      = tp / (tp + fp + fn + 1e-9)
    f1        = 2 * recall * precision / (precision + recall + 1e-9)
    print(f'Class: {class_name:12s}  Precision:{precision:.4f}  '
          f'Recall:{recall:.4f}  IOU:{iou1:.4f}  F1:{f1:.4f}')
    return iou1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate 3D-OutDet on LIDARDUSTX')

    parser.add_argument('-d', '--data_dir',
                        default='/home/bbb/dataset/data/LIDARDUSTX2')
    parser.add_argument('--label_config',
                        type=str,
                        default='binary_dedust.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/home/bbb/dataset/data/saved_models/dust/outdet.pt')
    parser.add_argument('-o', '--test_output_path',
                        default='/home/bbb/dataset/data/eval_results')
    parser.add_argument('--K',
                        type=int, default=3)
    parser.add_argument('--depth',
                        type=int, default=1,
                        help='模型深度，需与训练时一致（exp_011 填2，exp_012 填3）')
    parser.add_argument('--test_batch_size',
                        type=int, default=1)
    parser.add_argument('--device',
                        type=str, default='cuda:0')
    parser.add_argument('--save_pred',
                        action='store_true', default=False,
                        help='是否将预测标签保存为 .label 文件')
    parser.add_argument('--sensor',
                        type=str, default=None,
                        help='只评估指定传感器的帧，可选: ls64 ls128 ly150 ly300 m1 ouster')

    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)

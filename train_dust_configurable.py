#!/usr/bin/env python3
import os
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
import yaml
from tqdm import tqdm
from sklearn.metrics import multilabel_confusion_matrix
import warnings
import json

from dataset.utils.collate import collate_fn_cp
from deterministic import configure_randomness
from modules import OutDet
from modules.lovasz_losses import lovasz_softmax_flat
from dataset import WadsPointCloudDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main(args):
    data_path        = args.data_dir
    train_batch_size = args.train_batch_size
    val_batch_size   = args.val_batch_size
    model_save_path  = args.model_save_path
    output_model_path      = os.path.join(model_save_path, 'outdet.pt')
    output_loss_model_path = os.path.join(model_save_path, 'outdet_loss.pt')
    os.makedirs(model_save_path, exist_ok=True)
    device = torch.device(args.device)
    dilate = args.dilate

    # 保存实验配置到JSON
    config_save_path = os.path.join(model_save_path, 'experiment_config.json')

    # 读取 yaml 配置
    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)

    class_strings    = config["labels"]
    learning_map     = config["learning_map"]
    class_inv_remap  = config["learning_map_inv"]
    num_classes      = len(class_inv_remap)
    ordered_class_names = [class_strings[class_inv_remap[i]] for i in range(num_classes)]

    # 计算类别权重
    epsilon_w = args.epsilon_w
    content   = torch.zeros(num_classes, dtype=torch.float, device=device)
    for cl, freq in config["content"].items():
        x_cl = learning_map[cl]
        content[x_cl] += freq
    class_w = content / torch.sum(content)
    loss_w  = 1.0 / (class_w + epsilon_w)
    print("Loss weights from content: ", loss_w.data)

    # 模型
    model = OutDet(num_classes=num_classes, kernel_size=args.K,
                   depth=args.depth, dilate=dilate)
    model = model.to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                          weight_decay=args.weight_decay)

    # 学习率调度器
    if args.scheduler == 'multistep':
        milestones = [int(x) for x in args.milestones.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=milestones, gamma=args.gamma
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=args.num_epoch
        )
    else:
        scheduler = None

    # 损失函数
    criterion_ce = torch.nn.CrossEntropyLoss(weight=loss_w)
    criterion_focal = FocalLoss(alpha=loss_w, gamma=args.focal_gamma)

    # 数据集
    tree_k = int(np.round(args.K * args.K))
    train_dataset = WadsPointCloudDataset(
        device, data_path + '/sequences/', imageset='train',
        label_conf=args.label_config, k=tree_k, shuffle_indices=False
    )
    val_dataset = WadsPointCloudDataset(
        device, data_path + '/sequences/', imageset='val',
        label_conf=args.label_config, k=tree_k
    )

    train_dataset_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=train_batch_size,
        shuffle=True, num_workers=8, collate_fn=collate_fn_cp
    )
    val_dataset_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=val_batch_size,
        shuffle=False, num_workers=8, collate_fn=collate_fn_cp
    )

    # 保存实验配置
    exp_config = {
        "K": args.K,
        "tree_k": tree_k,
        "depth": args.depth,
        "dilate": dilate,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "milestones": args.milestones if args.scheduler == 'multistep' else None,
        "gamma": args.gamma if args.scheduler == 'multistep' else None,
        "num_epoch": args.num_epoch,
        "epsilon_w": epsilon_w,
        "loss_ce_weight": args.loss_ce_weight,
        "loss_lovasz_weight": args.loss_lovasz_weight,
        "loss_focal_weight": args.loss_focal_weight,
        "focal_gamma": args.focal_gamma,
        "train_batch_size": train_batch_size,
        "val_batch_size": val_batch_size,
    }
    with open(config_save_path, 'w') as f:
        json.dump(exp_config, f, indent=2)
    print(f"实验配置已保存至: {config_save_path}")

    # 训练循环
    epoch         = 0
    best_val_miou = 0
    best_val_loss = np.inf
    best_epoch_miou = 0
    best_epoch_loss = 0
    train_losses  = []
    val_losses    = []
    val_mious     = []

    while epoch < args.num_epoch:
        train_loss = train_epoch(
            epoch, model, train_dataset_loader, criterion_ce, criterion_focal,
            optimizer, num_classes, ordered_class_names, device, args
        )
        val_miou, val_loss = validate_epoch(
            epoch, model, val_dataset_loader, criterion_ce, criterion_focal,
            num_classes, ordered_class_names, device, args
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_mious.append(val_miou)

        if scheduler:
            scheduler.step()

        if best_val_miou <= val_miou:
            print(f'Saving model at epoch: {epoch}')
            best_val_miou = val_miou
            best_epoch_miou = epoch
            torch.save(model.state_dict(), output_model_path)
        if best_val_loss > val_loss:
            print(f'Saving loss model at epoch: {epoch}')
            best_val_loss = val_loss
            best_epoch_loss = epoch
            torch.save(model.state_dict(), output_loss_model_path)

        print('Epoch: %d  Current val miou: %.3f  Best val miou: %.3f' %
              (epoch, val_miou, best_val_miou))
        epoch += 1

    # 保存最后一个 epoch 的模型
    torch.save(model.state_dict(), os.path.join(model_save_path, 'outdet_last.pt'))

    # 保存训练信息
    training_info = {
        "best_val_miou": float(best_val_miou),
        "best_epoch_miou": int(best_epoch_miou),
        "best_val_loss": float(best_val_loss),
        "best_epoch_loss": int(best_epoch_loss),
        "final_train_loss": float(train_losses[-1]),
        "train_losses": [float(x) for x in train_losses],
        "val_losses": [float(x) for x in val_losses],
        "val_mious": [float(x) for x in val_mious],
    }
    training_info_path = os.path.join(model_save_path, 'training_info.json')
    with open(training_info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    print(f"训练信息已保存至: {training_info_path}")

    # 保存 loss 曲线图
    loss_curve_path = os.path.join(model_save_path, 'train_loss.png')
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.subplot(1, 3, 2)
    plt.plot(val_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.subplot(1, 3, 3)
    plt.plot(val_mious)
    plt.xlabel('Epoch')
    plt.ylabel('mIOU')
    plt.title('Validation mIOU')
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()
    print(f'曲线图已保存至: {loss_curve_path}')


def train_epoch(epoch, model, train_dataset_loader, criterion_ce, criterion_focal,
                optimizer, n_classes, class_names, device, args):
    loss_list = []
    pbar = tqdm(total=len(train_dataset_loader), desc=f'Epoch {epoch}')
    model.train()
    pcm = np.zeros(shape=(n_classes, 2, 2))

    for i_iter, batch in enumerate(train_dataset_loader):
        data  = batch['data'][0].to(device)
        ind   = batch['ind'][0]
        dist  = batch['dist'][0].to(device)
        label = batch['label'][0].long().to(device)

        optimizer.zero_grad()
        logit = model(data, dist, ind)

        # 组合损失函数
        loss = 0
        if args.loss_ce_weight > 0:
            loss += args.loss_ce_weight * criterion_ce(logit, label)
        if args.loss_lovasz_weight > 0:
            loss += args.loss_lovasz_weight * lovasz_softmax_flat(
                torch.nn.functional.softmax(logit, dim=1), label, ignore=None
            )
        if args.loss_focal_weight > 0:
            loss += args.loss_focal_weight * criterion_focal(logit, label)

        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())

        pbar.update(1)
        pbar.set_postfix({'Loss': np.mean(loss_list)})

        predict_labels = torch.argmax(logit, dim=-1)
        mcm = multilabel_confusion_matrix(
            y_true=label.cpu().numpy(),
            y_pred=predict_labels.cpu().numpy(),
            labels=[i for i in range(n_classes)]
        )
        pcm += mcm

    pbar.close()
    IOUs = [evaluate_cm(pcm[i], class_names[i]) for i in range(n_classes)]
    class_jaccard = torch.tensor(np.array(IOUs))
    print('Average iou: ', class_jaccard[1:].mean())
    return np.mean(loss_list)


def validate_epoch(epoch, model, val_dataset_loader, criterion_ce, criterion_focal,
                   n_classes, class_names, device, args):
    model.eval()
    val_loss_list = []
    pcm  = np.zeros(shape=(n_classes, 2, 2))
    pbar = tqdm(total=len(val_dataset_loader), desc=f'Epoch {epoch}')

    with torch.no_grad():
        for batch in val_dataset_loader:
            data  = batch['data'][0].to(device)
            ind   = batch['ind'][0]
            dist  = batch['dist'][0].to(device)
            label = batch['label'][0].long().to(device)

            logit = model(data, dist, ind)

            # 组合损失函数
            loss = 0
            if args.loss_ce_weight > 0:
                loss += args.loss_ce_weight * criterion_ce(logit, label)
            if args.loss_lovasz_weight > 0:
                loss += args.loss_lovasz_weight * lovasz_softmax_flat(
                    torch.nn.functional.softmax(logit, dim=1), label, ignore=None
                )
            if args.loss_focal_weight > 0:
                loss += args.loss_focal_weight * criterion_focal(logit, label)

            predict_labels = torch.argmax(logit, dim=1)
            mcm = multilabel_confusion_matrix(
                y_true=label.cpu().numpy(),
                y_pred=predict_labels.cpu().numpy(),
                labels=[i for i in range(n_classes)]
            )
            pcm += mcm
            val_loss_list.append(loss.detach().cpu().numpy())
            pbar.update()
            pbar.set_postfix({'Loss': np.mean(val_loss_list)})

    pbar.close()
    IOUs = [evaluate_cm(pcm[i], class_names[i]) for i in range(1, n_classes)]
    class_jaccard = torch.tensor(np.array(IOUs))
    val_miou = class_jaccard.sum().item() / class_jaccard.size(0)
    return val_miou, np.mean(val_loss_list)


def evaluate_cm(cm, class_name):
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    iou1      = tp / (tp + fp + fn + 1e-9)
    f1        = 2 * recall * precision / (precision + recall + 1e-9)
    print(f'Class: {class_name}, Precision:{precision:.4f}, '
          f'Recall:{recall:.4f}, IOU:{iou1:.4f}, F1:{f1:.4f}')
    return iou1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3D-OutDet on LIDARDUSTX (Configurable)')

    parser.add_argument('-d', '--data_dir',
                        default='/home/bbb/dataset/data/LIDARDUSTX2')
    parser.add_argument('--label_config',
                        type=str,
                        default='binary_dedust.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/home/bbb/dataset/data/saved_models/dust/')
    parser.add_argument('--K',
                        type=int, default=5)
    parser.add_argument('--depth',
                        type=int, default=1,
                        help='模型深度')
    parser.add_argument('--dilate',
                        type=int, default=1,
                        help='膨胀率')
    parser.add_argument('--train_batch_size',
                        type=int, default=1)
    parser.add_argument('--val_batch_size',
                        type=int, default=1)
    parser.add_argument('--device',
                        type=str, default='cuda:0')
    parser.add_argument('--num_epoch',
                        type=int, default=50)

    # 优化器参数
    parser.add_argument('--lr',
                        type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--weight_decay',
                        type=float, default=1e-4,
                        help='权重衰减')

    # 学习率调度器
    parser.add_argument('--scheduler',
                        type=str, default='multistep',
                        choices=['multistep', 'cosine', 'none'],
                        help='学习率调度器类型')
    parser.add_argument('--milestones',
                        type=str, default='30,40,45',
                        help='MultiStepLR的milestones，逗号分隔')
    parser.add_argument('--gamma',
                        type=float, default=0.1,
                        help='MultiStepLR的gamma')

    # 损失函数权重
    parser.add_argument('--loss_ce_weight',
                        type=float, default=1.0,
                        help='CrossEntropy损失权重')
    parser.add_argument('--loss_lovasz_weight',
                        type=float, default=1.0,
                        help='Lovász损失权重')
    parser.add_argument('--loss_focal_weight',
                        type=float, default=0.0,
                        help='Focal损失权重')
    parser.add_argument('--focal_gamma',
                        type=float, default=2.0,
                        help='Focal Loss的gamma参数')

    # 类别权重
    parser.add_argument('--epsilon_w',
                        type=float, default=1e-3,
                        help='计算类别权重时的epsilon')

    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)


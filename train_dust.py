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

from dataset.utils.collate import collate_fn_cp
from deterministic import configure_randomness
from modules import OutDet
from modules.lovasz_losses import lovasz_softmax_flat
from dataset import DustPointCloudDataset
import matplotlib
matplotlib.use('Agg')   # 服务器无显示器环境，不用 plt.show()
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def main(args):
    data_path        = args.data_dir
    train_batch_size = args.train_batch_size
    val_batch_size   = args.val_batch_size
    model_save_path  = args.model_save_path
    output_model_path      = os.path.join(model_save_path, 'outdet.pt')
    output_loss_model_path = os.path.join(model_save_path, 'outdet_loss.pt')
    os.makedirs(model_save_path, exist_ok=True)
    device = torch.device(args.device)
    dilate = 1

    # 读取 yaml 配置
    with open(args.label_config, 'r') as stream:
        config = yaml.safe_load(stream)

    class_strings    = config["labels"]
    learning_map     = config["learning_map"]
    class_inv_remap  = config["learning_map_inv"]
    num_classes      = len(class_inv_remap)
    ordered_class_names = [class_strings[class_inv_remap[i]] for i in range(num_classes)]

    # 计算类别权重（少数类 dust 会获得更大权重）
    epsilon_w = 1e-3
    content   = torch.zeros(num_classes, dtype=torch.float, device=device)
    for cl, freq in config["content"].items():
        x_cl = learning_map[cl]
        content[x_cl] += freq
    class_w = content / torch.sum(content)
    loss_w  = 1.0 / (class_w + epsilon_w)
    print("Loss weights from content: ", loss_w.data)

    # 模型
    model = OutDet(num_classes=num_classes, kernel_size=args.K, depth=1, dilate=dilate)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=[30, 40, 45], gamma=0.1
    )
    criterion = torch.nn.CrossEntropyLoss(weight=loss_w)

    # 数据集
    tree_k = int(np.round(args.K * args.K))
    train_dataset = DustPointCloudDataset(
        device, data_path + '/sequences/', imageset='train',
        label_conf=args.label_config, k=tree_k, shuffle_indices=False
    )
    val_dataset = DustPointCloudDataset(
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

    # 训练循环
    epoch         = 0
    best_val_miou = 0
    best_val_loss = np.inf
    train_losses  = []

    while epoch < args.num_epoch:
        train_loss = train_epoch(
            epoch, model, train_dataset_loader, criterion,
            optimizer, num_classes, ordered_class_names, device
        )
        val_miou, val_loss = validate_epoch(
            epoch, model, val_dataset_loader, criterion,
            num_classes, ordered_class_names, device
        )
        train_losses.append(train_loss)
        scheduler.step()

        if best_val_miou <= val_miou:
            print(f'Saving model at epoch: {epoch}')
            best_val_miou = val_miou
            torch.save(model.state_dict(), output_model_path)
        if best_val_loss > val_loss:
            print(f'Saving loss model at epoch: {epoch}')
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_loss_model_path)

        print('Epoch: %d  Current val miou: %.3f  Best val miou: %.3f' %
              (epoch, val_miou, best_val_miou))
        epoch += 1

    # 保存最后一个 epoch 的模型
    torch.save(model.state_dict(), os.path.join(model_save_path, 'outdet_last.pt'))

    # 保存 loss 曲线图（服务器环境不能 plt.show()，改为写文件）
    loss_curve_path = os.path.join(model_save_path, 'train_loss.png')
    plt.figure()
    plt.plot(train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig(loss_curve_path)
    plt.close()
    print(f'Loss 曲线已保存至: {loss_curve_path}')


def train_epoch(epoch, model, train_dataset_loader, criterion,
                optimizer, n_classes, class_names, device):
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
        loss  = criterion(logit, label) + lovasz_softmax_flat(
            torch.nn.functional.softmax(logit, dim=1), label, ignore=None
        )
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


def validate_epoch(epoch, model, val_dataset_loader, criterion,
                   n_classes, class_names, device):
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
            loss  = criterion(logit, label) + lovasz_softmax_flat(
                torch.nn.functional.softmax(logit, dim=1), label, ignore=None
            )
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
    parser = argparse.ArgumentParser(description='Train 3D-OutDet on LIDARDUSTX')

    parser.add_argument('-d', '--data_dir',
                        default='/home/bbb/dataset/data/LIDARDUSTX2')
    parser.add_argument('--label_config',
                        type=str,
                        default='binary_dedust.yaml')
    parser.add_argument('-p', '--model_save_path',
                        default='/home/bbb/dataset/data/saved_models/dust/')
    parser.add_argument('--K',
                        type=int, default=3)
    parser.add_argument('--train_batch_size',
                        type=int, default=1)
    parser.add_argument('--val_batch_size',
                        type=int, default=1)
    parser.add_argument('--device',
                        type=str, default='cuda:0')
    parser.add_argument('--num_epoch',
                        type=int, default=50)

    args = parser.parse_args()
    print(' '.join(sys.argv))
    print(args)
    torch.backends.cuda.matmul.allow_tf32 = True
    configure_randomness(12345)
    main(args)

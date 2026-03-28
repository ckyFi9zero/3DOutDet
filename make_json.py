#!/usr/bin/env python3
"""
make_json.py
把多次 eval_dust.py 的终端输出（拼在一起）解析并生成可导入实验记录表的 JSON。

用法（推荐，管道直接生成）：
    (
      python eval_dust.py ... --K 5
      for s in ls64 ls128 ly150 ly300 m1 ouster; do
        python eval_dust.py ... --K 5 --sensor $s
      done
    ) 2>&1 | python make_json.py --name "exp-01-K5" --out experiment_K5.json

或者粘贴输出：
    python make_json.py --name "exp-01-K5" --out experiment_K5.json
    （然后粘贴内容，Ctrl+D 结束）
"""

import re
import json
import sys
import argparse
import os
from datetime import datetime

SENSORS = ["ls64", "ls128", "ly150", "ly300", "m1", "ouster"]

def parse_dust_block(block):
    m = re.search(
        r'Class:\s*Dust\s+Precision:([\d.]+)\s+Recall:([\d.]+)\s+IOU:([\d.]+)\s+F1:([\d.]+)',
        block
    )
    if m:
        return {
            "precision": round(float(m.group(1)) * 100, 2),
            "recall":    round(float(m.group(2)) * 100, 2),
            "iou":       round(float(m.group(3)) * 100, 2),
            "f1":        round(float(m.group(4)) * 100, 2),
        }
    return None

def parse_miou(block):
    m = re.search(r'(\d+\.\d+)\s*&(\d+\.\d+)\s*&(\d+\.\d+)', block)
    if m:
        return round(float(m.group(3)), 2)
    return None

def extract_k(block):
    m = re.search(r"K=(\d+)", block)
    if m:
        return int(m.group(1))
    return None

def split_into_runs(text):
    """按「已加载模型」把整段输出切成多个独立的 eval 运行块"""
    parts = re.split(r'(?=已加载模型)', text)
    return [p.strip() for p in parts if p.strip()]

def identify_run(block):
    """判断是全部传感器还是某个单传感器"""
    m = re.search(r'仅评估\s*\[(\w+)\]', block)
    if m:
        return m.group(1).lower()
    return "all"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",  default="", help="实验名称")
    parser.add_argument("--desc",  default="", help="改动描述")
    parser.add_argument("--notes", default="", help="备注")
    parser.add_argument("--out",   default="experiment_result.json", help="输出路径")
    parser.add_argument("--exp_config", default=None, help="实验配置JSON文件路径")
    parser.add_argument("--training_info", default=None, help="训练信息JSON文件路径")
    parser.add_argument("--model_path", default="", help="模型保存路径")
    args = parser.parse_args()

    if sys.stdin.isatty():
        print("请粘贴 eval_dust.py 的终端输出，完成后按 Ctrl+D：")
        print("-" * 60)

    text = sys.stdin.read()
    if not text.strip():
        print("[ERROR] 没有读到任何内容")
        sys.exit(1)

    runs = split_into_runs(text)
    if not runs:
        print("[ERROR] 未能识别出任何评估块，请检查输入内容")
        sys.exit(1)

    print(f"识别到 {len(runs)} 个评估块，开始解析...")

    result = {}
    k_val = None

    for run in runs:
        label = identify_run(run)
        if k_val is None:
            k_val = extract_k(run)

        metrics = parse_dust_block(run)
        if not metrics:
            print(f"  [WARN] [{label}] 未找到 Dust 指标，跳过")
            continue

        if label == "all":
            result["all_iou"]       = metrics["iou"]
            result["all_recall"]    = metrics["recall"]
            result["all_precision"] = metrics["precision"]
            result["all_f1"]        = metrics["f1"]
            miou = parse_miou(run)
            if miou:
                result["all_miou"] = miou
            print(f"  [全部传感器] Dust IOU={metrics['iou']}%  mIOU={result.get('all_miou','?')}%")
        elif label in SENSORS:
            result[f"{label}_iou"] = metrics["iou"]
            print(f"  [{label:8s}] Dust IOU={metrics['iou']}%")
        else:
            print(f"  [WARN] 未知传感器标签：{label}，跳过")

    if not result:
        print("[ERROR] 解析失败，请确认粘贴的内容包含完整的 eval_dust.py 输出")
        sys.exit(1)

    k = k_val or "?"
    result["knn_k"]  = k
    result["tree_k"] = "?" if k == "?" else k * k
    result["id"]          = int(datetime.now().timestamp() * 1000)
    result["name"]        = args.name or f"exp-K{k}"
    result["date"]        = datetime.now().strftime("%Y-%m-%d")
    result["description"] = args.desc or f"K={k}, tree_k={result['tree_k']}"
    result["notes"]       = args.notes
    result["dataset"]     = "LIDARDUSTX2"
    result["sensors"]     = "ls64+ls128+ly150+ly300+m1+ouster"
    result["split"]       = "70/15/15 per-sensor stratified"
    result["model"]       = "3D-OutDet"
    result["depth"]       = 1
    result["extra_features"] = ""
    result["loss"]        = "CrossEntropy + Lovász"
    result["epochs"]      = 50
    result["lr"]          = "1e-3 → MultiStepLR [30,40,45] γ=0.1"

    # 加载实验配置
    if args.exp_config and os.path.exists(args.exp_config):
        with open(args.exp_config, 'r') as f:
            exp_config = json.load(f)
        result["experiment_config"] = exp_config
        # 更新基本信息
        result["depth"] = exp_config.get("depth", 1)
        result["epochs"] = exp_config.get("num_epoch", 50)

        # 构建loss描述
        loss_parts = []
        if exp_config.get("loss_ce_weight", 0) > 0:
            loss_parts.append(f"CE×{exp_config['loss_ce_weight']}")
        if exp_config.get("loss_lovasz_weight", 0) > 0:
            loss_parts.append(f"Lovász×{exp_config['loss_lovasz_weight']}")
        if exp_config.get("loss_focal_weight", 0) > 0:
            loss_parts.append(f"Focal×{exp_config['loss_focal_weight']}")
        result["loss"] = " + ".join(loss_parts) if loss_parts else "Unknown"

        # 构建lr描述
        if exp_config.get("scheduler") == "multistep":
            result["lr"] = f"{exp_config['lr']} → MultiStepLR {exp_config.get('milestones', '?')} γ={exp_config.get('gamma', '?')}"
        elif exp_config.get("scheduler") == "cosine":
            result["lr"] = f"{exp_config['lr']} → CosineAnnealing"
        else:
            result["lr"] = f"{exp_config['lr']} (constant)"

    # 加载训练信息
    if args.training_info and os.path.exists(args.training_info):
        with open(args.training_info, 'r') as f:
            training_info = json.load(f)
        result["training_info"] = training_info

    # 添加模型路径
    if args.model_path:
        result["model_path"] = args.model_path

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n── 汇总 ──────────────────────────────────────")
    print(f"  实验名称  : {result['name']}")
    print(f"  Dust IOU  : {result.get('all_iou', '未找到')}%")
    print(f"  Recall    : {result.get('all_recall', '未找到')}%")
    print(f"  Precision : {result.get('all_precision', '未找到')}%")
    print(f"  F1        : {result.get('all_f1', '未找到')}%")
    print(f"  mIOU      : {result.get('all_miou', '未找到')}%")
    print(f"\n  分传感器 Dust IOU：")
    for s in SENSORS:
        val = result.get(f"{s}_iou", "—")
        print(f"    {s:8s}: {val}%")
    print(f"\nJSON 已保存至：{args.out}")

if __name__ == "__main__":
    main()

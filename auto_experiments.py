#!/usr/bin/env python3
"""
自动化实验工作流
功能：自动执行多组参数的训练、评估、生成JSON
用法：python auto_experiments.py [--start_from EXP_ID] [--only EXP_ID]
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
import argparse

# 传感器列表
SENSORS = ["ls64", "ls128", "ly150", "ly300", "m1", "ouster"]


def load_config(config_file="experiments_config.json"):
    """加载实验配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def run_command(cmd, description=""):
    """运行命令并实时输出"""
    print(f"\n{'='*80}")
    print(f"执行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*80}\n")

    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)

    process.wait()
    if process.returncode != 0:
        print(f"\n[ERROR] 命令执行失败，返回码: {process.returncode}")
        return False, ''.join(output_lines)

    return True, ''.join(output_lines)


def build_train_command(base_config, exp_config, model_save_path):
    """构建训练命令"""
    params = exp_config['params']
    cmd_parts = [
        "python train_dust_configurable.py",
        f"-d {base_config['data_dir']}",
        f"--label_config {base_config['label_config']}",
        f"-p {model_save_path}",
        f"--K {base_config['K']}",
        f"--device {base_config['device']}",
        f"--train_batch_size {base_config['train_batch_size']}",
        f"--val_batch_size {base_config['val_batch_size']}",
    ]

    # 添加实验特定参数
    for key, value in params.items():
        cmd_parts.append(f"--{key} {value}")

    return " ".join(cmd_parts)


def build_eval_command(base_config, model_path, sensor=None):
    """构建评估命令"""
    cmd_parts = [
        "python eval_dust.py",
        f"-d {base_config['data_dir']}",
        f"--label_config {base_config['label_config']}",
        f"-p {model_path}",
        f"--K {base_config['K']}",
        f"--device {base_config['device']}",
        f"--test_batch_size 1",
    ]

    if sensor:
        cmd_parts.append(f"--sensor {sensor}")

    return " ".join(cmd_parts)


def run_experiment(base_config, exp_config, results_dir, skip_training=False):
    """运行单个实验"""
    exp_id = exp_config['id']
    exp_name = exp_config['name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n\n{'#'*80}")
    print(f"# 开始实验: {exp_id}")
    print(f"# 名称: {exp_name}")
    print(f"# 时间: {timestamp}")
    print(f"{'#'*80}\n")

    # 创建实验目录
    exp_dir = os.path.join(results_dir, f"{exp_id}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    model_save_path = os.path.join(exp_dir, "models")
    os.makedirs(model_save_path, exist_ok=True)

    # 1. 训练
    if not skip_training:
        train_cmd = build_train_command(base_config, exp_config, model_save_path)
        success, train_output = run_command(train_cmd, "训练模型")

        if not success:
            print(f"\n[ERROR] 实验 {exp_id} 训练失败，跳过")
            return False

        print(f"\n[SUCCESS] 训练完成")
    else:
        print(f"\n[SKIP] 跳过训练")

    # 2. 评估
    model_path = os.path.join(model_save_path, "outdet.pt")
    if not os.path.exists(model_path):
        print(f"\n[ERROR] 模型文件不存在: {model_path}")
        return False

    eval_outputs = []

    # 评估全部传感器
    eval_cmd = build_eval_command(base_config, model_path)
    success, eval_output = run_command(eval_cmd, "评估全部传感器")
    if success:
        eval_outputs.append(eval_output)
    else:
        print(f"\n[ERROR] 评估失败")
        return False

    # 评估各个传感器
    for sensor in SENSORS:
        eval_cmd = build_eval_command(base_config, model_path, sensor)
        success, eval_output = run_command(eval_cmd, f"评估传感器 {sensor}")
        if success:
            eval_outputs.append(eval_output)
        else:
            print(f"\n[WARNING] 传感器 {sensor} 评估失败")

    # 3. 生成JSON
    combined_eval_output = "\n".join(eval_outputs)
    json_output_path = os.path.join(exp_dir, f"{exp_id}_{timestamp}.json")

    # 构建make_json命令
    exp_config_path = os.path.join(model_save_path, "experiment_config.json")
    training_info_path = os.path.join(model_save_path, "training_info.json")

    make_json_cmd = [
        "python make_json.py",
        f"--name '{exp_name}'",
        f"--desc '{exp_config['description']}'",
        f"--notes '{exp_config['modifications']}'",
        f"--out {json_output_path}",
    ]

    if os.path.exists(exp_config_path):
        make_json_cmd.append(f"--exp_config {exp_config_path}")
    if os.path.exists(training_info_path):
        make_json_cmd.append(f"--training_info {training_info_path}")

    make_json_cmd.append(f"--model_path {model_path}")

    make_json_full_cmd = " ".join(make_json_cmd)

    # 通过管道传入评估输出
    process = subprocess.Popen(
        make_json_full_cmd, shell=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate(input=combined_eval_output)

    if process.returncode != 0:
        print(f"\n[ERROR] 生成JSON失败")
        print(stderr)
        return False

    print(stdout)
    print(f"\n[SUCCESS] JSON已保存至: {json_output_path}")

    # 保存实验摘要
    summary = {
        "experiment_id": exp_id,
        "name": exp_name,
        "timestamp": timestamp,
        "status": "completed",
        "model_path": model_path,
        "json_path": json_output_path,
        "experiment_dir": exp_dir,
    }

    summary_path = os.path.join(exp_dir, "summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'#'*80}")
    print(f"# 实验 {exp_id} 完成")
    print(f"# 结果目录: {exp_dir}")
    print(f"{'#'*80}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description='自动化实验工作流')
    parser.add_argument('--config', default='experiments_config.json',
                       help='实验配置文件路径')
    parser.add_argument('--results_dir', default='experiment_results',
                       help='结果保存目录')
    parser.add_argument('--start_from', default=None,
                       help='从指定实验ID开始（跳过之前的实验）')
    parser.add_argument('--only', default=None,
                       help='只运行指定的实验ID')
    parser.add_argument('--skip_training', action='store_true',
                       help='跳过训练，只进行评估（需要模型已存在）')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    base_config = config['base_config']
    experiments = config['experiments']

    # 创建结果目录
    os.makedirs(args.results_dir, exist_ok=True)

    # 过滤实验
    if args.only:
        experiments = [exp for exp in experiments if exp['id'] == args.only]
        if not experiments:
            print(f"[ERROR] 未找到实验ID: {args.only}")
            return
    elif args.start_from:
        found = False
        filtered_experiments = []
        for exp in experiments:
            if exp['id'] == args.start_from:
                found = True
            if found:
                filtered_experiments.append(exp)
        if not found:
            print(f"[ERROR] 未找到起始实验ID: {args.start_from}")
            return
        experiments = filtered_experiments

    print(f"\n总共将运行 {len(experiments)} 个实验\n")

    # 运行实验
    success_count = 0
    failed_experiments = []

    for i, exp_config in enumerate(experiments, 1):
        print(f"\n\n{'='*80}")
        print(f"进度: {i}/{len(experiments)}")
        print(f"{'='*80}")

        success = run_experiment(
            base_config, exp_config, args.results_dir,
            skip_training=args.skip_training
        )

        if success:
            success_count += 1
        else:
            failed_experiments.append(exp_config['id'])

        # 每个实验之间暂停一下
        if i < len(experiments):
            print(f"\n等待5秒后开始下一个实验...")
            time.sleep(5)

    # 打印总结
    print(f"\n\n{'='*80}")
    print(f"所有实验完成")
    print(f"{'='*80}")
    print(f"成功: {success_count}/{len(experiments)}")
    if failed_experiments:
        print(f"失败的实验: {', '.join(failed_experiments)}")
    print(f"结果保存在: {args.results_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

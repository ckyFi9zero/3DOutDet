#!/usr/bin/env python3
"""
实验结果对比工具
读取所有实验的JSON文件，生成对比表格
"""

import json
import os
import glob
from datetime import datetime


def load_experiment_results(results_dir):
    """加载所有实验结果"""
    results = []

    # 查找所有实验目录
    exp_dirs = glob.glob(os.path.join(results_dir, "exp_*"))

    for exp_dir in sorted(exp_dirs):
        # 查找JSON文件
        json_files = glob.glob(os.path.join(exp_dir, "*.json"))
        json_files = [f for f in json_files if not f.endswith('summary.json')
                     and not f.endswith('experiment_config.json')
                     and not f.endswith('training_info.json')]

        if json_files:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                result = json.load(f)
                results.append(result)

    return results


def print_comparison_table(results):
    """打印对比表格"""
    if not results:
        print("没有找到实验结果")
        return

    print("\n" + "="*120)
    print("实验结果对比表")
    print("="*120)

    # 表头
    header = f"{'ID':<25} {'名称':<20} {'Dust IOU':<10} {'Recall':<10} {'Precision':<10} {'F1':<10} {'mIOU':<10}"
    print(header)
    print("-"*120)

    # 数据行
    for result in results:
        exp_id = result.get('id', result.get('experiment_id', 'N/A'))
        name = result.get('name', 'N/A')[:18]
        dust_iou = result.get('all_iou', 'N/A')
        recall = result.get('all_recall', 'N/A')
        precision = result.get('all_precision', 'N/A')
        f1 = result.get('all_f1', 'N/A')
        miou = result.get('all_miou', 'N/A')

        print(f"{exp_id:<25} {name:<20} {dust_iou:<10} {recall:<10} {precision:<10} {f1:<10} {miou:<10}")

    print("="*120)

    # 找出最佳结果
    best_iou = max(results, key=lambda x: x.get('all_iou', 0))
    best_recall = max(results, key=lambda x: x.get('all_recall', 0))
    best_precision = max(results, key=lambda x: x.get('all_precision', 0))

    print(f"\n最佳 Dust IOU: {best_iou.get('all_iou')}% - {best_iou.get('name')}")
    print(f"最佳 Recall: {best_recall.get('all_recall')}% - {best_recall.get('name')}")
    print(f"最佳 Precision: {best_precision.get('all_precision')}% - {best_precision.get('name')}")


def print_sensor_comparison(results):
    """打印传感器对比"""
    sensors = ["ls64", "ls128", "ly150", "ly300", "m1", "ouster"]

    print("\n" + "="*120)
    print("各传感器 Dust IOU 对比")
    print("="*120)

    # 表头
    header = f"{'实验ID':<25} " + " ".join([f"{s:<10}" for s in sensors])
    print(header)
    print("-"*120)

    # 数据行
    for result in results:
        exp_id = result.get('id', result.get('experiment_id', 'N/A'))
        row = f"{exp_id:<25} "
        for sensor in sensors:
            iou = result.get(f"{sensor}_iou", "N/A")
            row += f"{iou:<10} "
        print(row)

    print("="*120)


def export_to_csv(results, output_file="experiment_comparison.csv"):
    """导出为CSV文件"""
    import csv

    if not results:
        return

    sensors = ["ls64", "ls128", "ly150", "ly300", "m1", "ouster"]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['experiment_id', 'name', 'description', 'modifications',
                     'all_iou', 'all_recall', 'all_precision', 'all_f1', 'all_miou']
        fieldnames += [f"{s}_iou" for s in sensors]
        fieldnames += ['loss', 'lr', 'epochs', 'depth']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                'experiment_id': result.get('id', result.get('experiment_id', '')),
                'name': result.get('name', ''),
                'description': result.get('description', ''),
                'modifications': result.get('modifications', result.get('notes', '')),
                'all_iou': result.get('all_iou', ''),
                'all_recall': result.get('all_recall', ''),
                'all_precision': result.get('all_precision', ''),
                'all_f1': result.get('all_f1', ''),
                'all_miou': result.get('all_miou', ''),
                'loss': result.get('loss', ''),
                'lr': result.get('lr', ''),
                'epochs': result.get('epochs', ''),
                'depth': result.get('depth', ''),
            }

            for sensor in sensors:
                row[f"{sensor}_iou"] = result.get(f"{sensor}_iou", '')

            writer.writerow(row)

    print(f"\n结果已导出至: {output_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='实验结果对比工具')
    parser.add_argument('--results_dir', default='experiment_results',
                       help='实验结果目录')
    parser.add_argument('--csv', action='store_true',
                       help='导出为CSV文件')
    args = parser.parse_args()

    results = load_experiment_results(args.results_dir)

    if not results:
        print(f"在 {args.results_dir} 中没有找到实验结果")
    else:
        print(f"\n找到 {len(results)} 个实验结果\n")
        print_comparison_table(results)
        print_sensor_comparison(results)

        if args.csv:
            export_to_csv(results)

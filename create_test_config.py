#!/usr/bin/env python3
"""
快速测试脚本 - 验证自动化实验系统是否正常工作
只运行1个epoch来快速测试
"""

import json
import os

# 创建测试配置
test_config = {
    "base_config": {
        "data_dir": "/home/bbb/dataset/data/LIDARDUSTX2",
        "label_config": "binary_dedust.yaml",
        "K": 5,
        "device": "cuda:0",
        "train_batch_size": 1,
        "val_batch_size": 1
    },
    "experiments": [
        {
            "id": "test_001_quick",
            "name": "快速测试",
            "description": "1个epoch快速测试系统",
            "modifications": "仅用于测试，num_epoch=1",
            "params": {
                "num_epoch": 1,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "scheduler": "multistep",
                "milestones": "30,40,45",
                "gamma": 0.1,
                "loss_ce_weight": 1.0,
                "loss_lovasz_weight": 1.0,
                "loss_focal_weight": 0.0,
                "focal_gamma": 2.0,
                "epsilon_w": 1e-3,
                "depth": 1,
                "dilate": 1
            }
        }
    ]
}

# 保存测试配置
with open('test_config.json', 'w', encoding='utf-8') as f:
    json.dump(test_config, f, indent=2, ensure_ascii=False)

print("测试配置已创建: test_config.json")
print("\n运行测试:")
print("python auto_experiments.py --config test_config.json --results_dir test_results")

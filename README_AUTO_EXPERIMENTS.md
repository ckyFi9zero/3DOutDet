# 自动化实验系统使用说明

## 文件说明

### 1. `train_dust_configurable.py`
可配置的训练脚本，支持通过命令行参数调整：
- 损失函数权重（CE、Lovász、Focal）
- 学习率和调度器
- 模型深度
- 类别权重epsilon
- 等等

### 2. `experiments_config.json`
实验配置文件，定义了12组实验：
- exp_001: Baseline（基线）
- exp_002-003: Loss权重调整
- exp_004-005: 添加Focal Loss
- exp_006-008: 学习率策略调整
- exp_009-010: epsilon_w调整
- exp_011-012: 模型深度调整

### 3. `auto_experiments.py`
自动化工作流脚本，自动执行：训练 → 评估全部 → 评估6个传感器 → 生成JSON

### 4. `make_json.py`（已增强）
解析评估输出，生成包含完整实验配置的JSON文件

---

## 使用方法

### 方式1：运行所有实验
```bash
python auto_experiments.py
```

### 方式2：只运行某个实验
```bash
python auto_experiments.py --only exp_004_add_focal
```

### 方式3：从某个实验开始运行
```bash
# 从exp_006开始，运行exp_006到exp_012
python auto_experiments.py --start_from exp_006_lr_aggressive
```

### 方式4：自定义结果目录
```bash
python auto_experiments.py --results_dir my_experiments
```

---

## 输出结构

每个实验会创建独立的目录：
```
experiment_results/
├── exp_001_baseline_20260328_143025/
│   ├── models/
│   │   ├── outdet.pt              # 最佳mIOU模型
│   │   ├── outdet_loss.pt         # 最佳loss模型
│   │   ├── outdet_last.pt         # 最后epoch模型
│   │   ├── experiment_config.json # 实验配置
│   │   ├── training_info.json     # 训练信息
│   │   └── train_loss.png         # 训练曲线
│   ├── exp_001_baseline_20260328_143025.json  # 完整结果JSON
│   └── summary.json               # 实验摘要
├── exp_002_loss_weight_0.5_1_20260328_150130/
│   └── ...
└── ...
```

---

## JSON文件内容

生成的JSON文件包含：

```json
{
  "experiment_id": "exp_004_add_focal",
  "name": "添加Focal Loss",
  "description": "CE + Lovász + Focal (1:1:1)",
  "modifications": "添加Focal Loss，权重1.0",

  "experiment_config": {
    "loss_ce_weight": 1.0,
    "loss_lovasz_weight": 1.0,
    "loss_focal_weight": 1.0,
    "lr": 0.001,
    "scheduler": "multistep",
    ...
  },

  "training_info": {
    "best_val_miou": 0.789,
    "best_epoch_miou": 45,
    "train_losses": [...],
    "val_mious": [...]
  },

  "results": {
    "all_iou": 59.03,
    "all_recall": 72.32,
    "ls64_iou": 55.8,
    ...
  },

  "model_path": "/path/to/model"
}
```

---

## 修改实验配置

编辑 `experiments_config.json`：

```json
{
  "id": "exp_013_my_experiment",
  "name": "我的实验",
  "description": "实验描述",
  "modifications": "修改了什么",
  "params": {
    "num_epoch": 50,
    "lr": 1e-3,
    "loss_ce_weight": 1.0,
    "loss_lovasz_weight": 1.0,
    "loss_focal_weight": 0.5,
    ...
  }
}
```

然后运行：
```bash
python auto_experiments.py --only exp_013_my_experiment
```

---

## 注意事项

1. **K值固定为5**：在 `experiments_config.json` 的 `base_config` 中设置

2. **每个实验独立保存**：不会覆盖之前的结果

3. **自动命名**：JSON文件名包含实验ID和时间戳，确保唯一

4. **失败处理**：某个实验失败不会影响后续实验

5. **中断恢复**：可以使用 `--start_from` 从中断处继续

---

## 快速开始

1. 检查配置：
```bash
cat experiments_config.json
```

2. 测试单个实验：
```bash
python auto_experiments.py --only exp_001_baseline
```

3. 运行所有实验：
```bash
nohup python auto_experiments.py > experiments.log 2>&1 &
```

4. 查看进度：
```bash
tail -f experiments.log
```

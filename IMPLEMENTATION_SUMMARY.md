# 自动化实验系统 - 完整实现

## 已创建的文件

### 核心文件
1. **train_dust_configurable.py** - 可配置的训练脚本
   - 支持调整损失函数权重（CE、Lovász、Focal）
   - 支持多种学习率调度器（MultiStepLR、CosineAnnealing）
   - 支持调整模型深度、epsilon_w等参数
   - 自动保存实验配置和训练信息

2. **auto_experiments.py** - 自动化工作流脚本
   - 自动执行：训练 → 评估全部 → 评估6个传感器 → 生成JSON
   - 支持从指定实验开始运行
   - 支持只运行某个实验
   - 每个实验独立保存，不会覆盖

3. **experiments_config.json** - 实验配置文件
   - 定义了12组实验（按优先级排序）
   - 包含基线、loss调整、学习率策略、模型深度等

4. **make_json.py** (已增强) - 结果生成脚本
   - 自动读取实验配置和训练信息
   - 生成包含完整修改记录的JSON文件

### 辅助工具
5. **compare_experiments.py** - 实验对比工具
   - 生成对比表格
   - 显示各传感器结果
   - 导出CSV文件

6. **create_test_config.py** - 快速测试工具
   - 创建1个epoch的测试配置
   - 用于验证系统是否正常

7. **README_AUTO_EXPERIMENTS.md** - 使用说明文档

---

## 12组实验设计

### 高优先级（Loss相关）
1. **exp_001_baseline** - 基线实验
2. **exp_002_loss_weight_0.5_1** - CE:Lovász = 0.5:1
3. **exp_003_loss_weight_2_1** - CE:Lovász = 2:1
4. **exp_004_add_focal** - CE + Lovász + Focal (1:1:1)
5. **exp_005_focal_lovasz** - Focal + Lovász

### 高优先级（学习率相关）
6. **exp_006_lr_aggressive** - 激进学习率 (2e-3, [20,35,45])
7. **exp_007_lr_conservative** - 保守学习率 (5e-4, 100 epochs)
8. **exp_008_cosine_annealing** - 余弦退火调度器

### 中优先级（类别权重）
9. **exp_009_epsilon_large** - epsilon_w = 1e-2
10. **exp_010_epsilon_small** - epsilon_w = 1e-4

### 中优先级（模型深度）
11. **exp_011_depth_2** - depth = 2
12. **exp_012_depth_3** - depth = 3

---

## 使用流程

### 1. 快速测试（推荐先做）
```bash
# 创建测试配置
python create_test_config.py

# 运行测试（只训练1个epoch）
python auto_experiments.py --config test_config.json --results_dir test_results

# 检查测试结果
ls test_results/
```

### 2. 运行单个实验
```bash
# 运行baseline实验
python auto_experiments.py --only exp_001_baseline

# 运行添加Focal Loss的实验
python auto_experiments.py --only exp_004_add_focal
```

### 3. 运行所有实验（后台运行）
```bash
# 后台运行所有实验
nohup python auto_experiments.py > experiments.log 2>&1 &

# 查看进度
tail -f experiments.log

# 查看进程
ps aux | grep auto_experiments
```

### 4. 从中断处继续
```bash
# 如果实验中断，从exp_006继续
python auto_experiments.py --start_from exp_006_lr_aggressive
```

### 5. 对比结果
```bash
# 查看对比表格
python compare_experiments.py

# 导出CSV
python compare_experiments.py --csv

# 查看CSV
cat experiment_comparison.csv
```

---

## 输出结构

```
experiment_results/
├── exp_001_baseline_20260328_143025/
│   ├── models/
│   │   ├── outdet.pt              # 最佳mIOU模型
│   │   ├── outdet_loss.pt         # 最佳loss模型
│   │   ├── outdet_last.pt         # 最后epoch模型
│   │   ├── experiment_config.json # 实验配置
│   │   ├── training_info.json     # 训练信息（loss曲线等）
│   │   └── train_loss.png         # 训练曲线图
│   ├── exp_001_baseline_20260328_143025.json  # 完整结果
│   └── summary.json               # 实验摘要
├── exp_002_loss_weight_0.5_1_20260328_150130/
└── ...
```

---

## JSON文件示例

```json
{
  "id": 1774696738837,
  "name": "添加Focal Loss",
  "date": "2026-03-28",
  "description": "CE + Lovász + Focal (1:1:1)",
  "notes": "添加Focal Loss，权重1.0",

  "experiment_config": {
    "K": 5,
    "tree_k": 25,
    "depth": 1,
    "lr": 0.001,
    "loss_ce_weight": 1.0,
    "loss_lovasz_weight": 1.0,
    "loss_focal_weight": 1.0,
    "focal_gamma": 2.0,
    "epsilon_w": 0.001,
    "scheduler": "multistep",
    "milestones": "30,40,45",
    "gamma": 0.1,
    "num_epoch": 50
  },

  "training_info": {
    "best_val_miou": 0.789,
    "best_epoch_miou": 45,
    "best_val_loss": 0.123,
    "best_epoch_loss": 42,
    "final_train_loss": 0.098,
    "train_losses": [...],
    "val_losses": [...],
    "val_mious": [...]
  },

  "all_iou": 59.03,
  "all_recall": 72.32,
  "all_precision": 76.26,
  "all_f1": 74.24,
  "all_miou": 76.94,

  "ls64_iou": 55.8,
  "ls128_iou": 73.82,
  "ly150_iou": 79.36,
  "ly300_iou": 33.97,
  "m1_iou": 63.25,
  "ouster_iou": 76.41,

  "model_path": "/path/to/model",
  "loss": "CE×1.0 + Lovász×1.0 + Focal×1.0",
  "lr": "0.001 → MultiStepLR [30,40,45] γ=0.1"
}
```

---

## 特点

✅ **完全自动化** - 一键运行多组实验
✅ **不会覆盖** - 每个实验独立保存，带时间戳
✅ **完整记录** - JSON包含所有修改信息
✅ **可中断恢复** - 支持从指定实验继续
✅ **结果对比** - 自动生成对比表格
✅ **灵活配置** - 通过JSON文件轻松添加新实验

---

## 预计时间

假设每个实验训练50个epoch需要2小时：
- 单个实验：约2.5小时（训练2h + 评估0.5h）
- 12个实验：约30小时（可后台运行）
- exp_007（100 epochs）：约5小时

建议：先运行快速测试，确认无误后再运行全部实验。

---

## 下一步

1. 先运行快速测试验证系统
2. 运行1-2个完整实验检查结果
3. 确认无误后运行所有实验
4. 使用compare_experiments.py分析结果
5. 根据结果调整配置，添加新实验

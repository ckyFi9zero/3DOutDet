# 自动化实验系统 - 使用检查清单

## ✅ 系统已创建的文件

### 核心脚本
- [x] train_dust_configurable.py - 可配置训练脚本
- [x] auto_experiments.py - 自动化工作流
- [x] experiments_config.json - 12组实验配置
- [x] make_json.py (已增强) - 结果生成

### 辅助工具
- [x] compare_experiments.py - 结果对比工具
- [x] create_test_config.py - 快速测试工具
- [x] start_experiments.sh - 交互式启动脚本

### 文档
- [x] README_AUTO_EXPERIMENTS.md - 使用说明
- [x] IMPLEMENTATION_SUMMARY.md - 实现总结
- [x] CHECKLIST.md - 本检查清单

---

## 📋 使用前检查

### 1. 环境检查
```bash
# 检查Python环境
python --version  # 应该是Python 3.x

# 检查必要的包
python -c "import torch; print(torch.__version__)"
python -c "import yaml; print('yaml OK')"
python -c "import sklearn; print('sklearn OK')"
```

### 2. 数据检查
```bash
# 检查数据目录
ls /home/bbb/dataset/data/LIDARDUSTX2/sequences/

# 检查配置文件
cat binary_dedust.yaml
```

### 3. 文件权限
```bash
# 确保脚本可执行
chmod +x train_dust_configurable.py
chmod +x auto_experiments.py
chmod +x compare_experiments.py
chmod +x start_experiments.sh
```

---

## 🚀 快速开始（推荐步骤）

### 步骤1：快速测试（必做）
```bash
# 方式1：使用交互式脚本
./start_experiments.sh
# 选择选项 1

# 方式2：直接运行
python create_test_config.py
python auto_experiments.py --config test_config.json --results_dir test_results
```

**预期结果：**
- 训练1个epoch（约2-3分钟）
- 生成test_results目录
- 包含完整的JSON文件

### 步骤2：检查测试结果
```bash
# 查看生成的文件
ls -R test_results/

# 查看JSON内容
cat test_results/test_001_quick_*/test_001_quick_*.json | python -m json.tool | head -50
```

### 步骤3：运行单个完整实验
```bash
# 运行baseline实验
python auto_experiments.py --only exp_001_baseline

# 或使用交互式脚本
./start_experiments.sh
# 选择选项 2，输入 exp_001_baseline
```

**预期时间：** 约2.5小时

### 步骤4：检查完整实验结果
```bash
# 查看结果
ls experiment_results/exp_001_baseline_*/

# 查看JSON
cat experiment_results/exp_001_baseline_*/exp_001_baseline_*.json | python -m json.tool
```

### 步骤5：运行所有实验（可选）
```bash
# 后台运行
nohup python auto_experiments.py > experiments.log 2>&1 &

# 查看进度
tail -f experiments.log

# 或使用交互式脚本
./start_experiments.sh
# 选择选项 3
```

**预期时间：** 约30小时（12个实验）

---

## 📊 结果分析

### 查看对比表格
```bash
python compare_experiments.py
```

### 导出CSV
```bash
python compare_experiments.py --csv
```

### 查看特定实验
```bash
# 查看某个实验的详细信息
cat experiment_results/exp_004_add_focal_*/exp_004_add_focal_*.json | python -m json.tool
```

---

## 🔧 常见问题

### Q1: 训练中断了怎么办？
```bash
# 从中断的实验继续
python auto_experiments.py --start_from exp_006_lr_aggressive
```

### Q2: 想修改实验配置？
```bash
# 编辑配置文件
vim experiments_config.json

# 运行修改后的实验
python auto_experiments.py --only exp_013_my_experiment
```

### Q3: 想添加新实验？
在 experiments_config.json 的 experiments 数组中添加：
```json
{
  "id": "exp_013_my_test",
  "name": "我的测试",
  "description": "测试描述",
  "modifications": "修改了什么",
  "params": {
    "num_epoch": 50,
    "lr": 1e-3,
    ...
  }
}
```

### Q4: 如何查看训练曲线？
```bash
# 训练曲线图保存在
ls experiment_results/exp_*/models/train_loss.png

# 可以下载到本地查看
```

### Q5: 磁盘空间不够？
```bash
# 每个实验约占用几GB空间
# 可以删除不需要的模型文件
rm experiment_results/exp_*/models/outdet_last.pt
rm experiment_results/exp_*/models/outdet_loss.pt
```

---

## 📈 实验优先级建议

### 第一批（高优先级，Loss相关）
1. exp_001_baseline
2. exp_002_loss_weight_0.5_1
3. exp_003_loss_weight_2_1
4. exp_004_add_focal
5. exp_005_focal_lovasz

### 第二批（学习率相关）
6. exp_006_lr_aggressive
7. exp_008_cosine_annealing

### 第三批（其他）
8. exp_009_epsilon_large
9. exp_011_depth_2

---

## ✨ 完成后

1. 使用 compare_experiments.py 查看所有结果
2. 找出最佳配置
3. 根据结果设计新的实验
4. 重复优化过程

---

## 📞 需要帮助？

查看文档：
- README_AUTO_EXPERIMENTS.md - 详细使用说明
- IMPLEMENTATION_SUMMARY.md - 系统实现总结

或者直接问我！

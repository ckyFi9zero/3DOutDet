#!/bin/bash
# 快速启动脚本

echo "=========================================="
echo "自动化实验系统 - 快速启动"
echo "=========================================="
echo ""

# 检查必要文件
echo "检查文件..."
files=(
    "train_dust_configurable.py"
    "auto_experiments.py"
    "experiments_config.json"
    "eval_dust.py"
    "make_json.py"
)

all_exist=true
for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ 缺少文件: $file"
        all_exist=false
    else
        echo "✓ $file"
    fi
done

if [ "$all_exist" = false ]; then
    echo ""
    echo "错误：缺少必要文件"
    exit 1
fi

echo ""
echo "所有文件检查通过！"
echo ""
echo "=========================================="
echo "请选择操作："
echo "=========================================="
echo "1. 快速测试（1个epoch，验证系统）"
echo "2. 运行单个实验"
echo "3. 运行所有实验（后台）"
echo "4. 查看实验配置"
echo "5. 对比实验结果"
echo "6. 退出"
echo ""
read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "创建测试配置..."
        python create_test_config.py
        echo ""
        echo "开始快速测试..."
        python auto_experiments.py --config test_config.json --results_dir test_results
        ;;
    2)
        echo ""
        echo "可用的实验："
        python -c "
import json
with open('experiments_config.json', 'r') as f:
    config = json.load(f)
for i, exp in enumerate(config['experiments'], 1):
    print(f\"{i:2d}. {exp['id']:30s} - {exp['name']}\")
"
        echo ""
        read -p "请输入实验ID: " exp_id
        echo ""
        echo "运行实验: $exp_id"
        python auto_experiments.py --only "$exp_id"
        ;;
    3)
        echo ""
        read -p "确认要运行所有12个实验吗？这可能需要30小时 (y/n): " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo ""
            echo "后台运行所有实验..."
            nohup python auto_experiments.py > experiments.log 2>&1 &
            pid=$!
            echo "进程ID: $pid"
            echo "日志文件: experiments.log"
            echo ""
            echo "查看进度: tail -f experiments.log"
            echo "停止实验: kill $pid"
        else
            echo "已取消"
        fi
        ;;
    4)
        echo ""
        echo "实验配置列表："
        python -c "
import json
with open('experiments_config.json', 'r') as f:
    config = json.load(f)
print(f\"总共 {len(config['experiments'])} 个实验:\n\")
for i, exp in enumerate(config['experiments'], 1):
    print(f\"{i:2d}. {exp['id']}\")
    print(f\"    名称: {exp['name']}\")
    print(f\"    描述: {exp['description']}\")
    print(f\"    修改: {exp['modifications']}\")
    print()
"
        ;;
    5)
        echo ""
        if [ -d "experiment_results" ]; then
            python compare_experiments.py
            echo ""
            read -p "是否导出CSV? (y/n): " export_csv
            if [ "$export_csv" = "y" ] || [ "$export_csv" = "Y" ]; then
                python compare_experiments.py --csv
            fi
        else
            echo "还没有实验结果"
        fi
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "完成！"

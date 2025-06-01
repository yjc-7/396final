# 快速开始指南

这是声音分类系统的快速开始指南，帮助你在几分钟内运行整个系统。

## 🚀 真实数据一键开始（推荐）

如果你已经有真实的音频数据：

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键运行完整流程
py quick_start_real_data.py
```

这个脚本会自动：
- 检查和组织你的数据
- 训练针对真实数据优化的模型
- 生成预测结果和可视化

## 📋 手动分步骤运行

### 步骤1: 准备真实数据

将你的音频文件放入对应目录（支持多种命名格式）：

```
data/
├── laughing/              # 或 laughing-sounds/
│   ├── real_laugh_1.wav
│   └── real_laugh_2.mp3
├── sighing/               # 或 sighing-sounds/
│   ├── real_sigh_1.wav
│   └── real_sigh_2.flac
├── tongue-clicking/       # 或 tongue_clicking/
├── throat-clearing/       # 或 throat_clearing/
├── teeth-grinding/        # 或 teeth_grinding/
├── yawning/
└── lip-smacking/          # 或 lip_smacking/
```

**支持格式**: WAV, MP3, FLAC, M4A, OGG

### 步骤2: 组织和分析数据
```bash
# 自动重命名目录并分析数据分布
py organize_data.py

# 查看数据统计
py organize_data.py --analyze_only
```

输出示例：
```
📊 Data distribution analysis:
  Total files: 313
  Real data files: 313
  Synthetic data files: 0

  Class distribution:
    laughing: 25 files (8.0%)
    sighing: 51 files (16.3%)
    tongue_clicking: 47 files (15.0%)
    throat_clearing: 54 files (17.3%)
    teeth_grinding: 43 files (13.7%)
    yawning: 48 files (15.3%)
    lip_smacking: 45 files (14.4%)

  ✓ Dataset is reasonably balanced (ratio: 2.2)
```

### 步骤3: 训练模型

#### 真实数据训练（推荐）
```bash
# 使用优化的真实数据训练
py train_real_data.py

# 自定义参数
py train_real_data.py --epochs 30 --batch_size 16 --learning_rate 1e-4

# 如果AST模型有问题，使用CNN
py train_real_data.py --model_type cnn
```

#### 传统训练方式
```bash
py train.py
```

#### 合成数据训练（测试用）
```bash
# 先生成合成数据
py data_generator.py

# 再训练
py train.py
```

### 步骤4: 测试推理
```bash
# 使用真实数据训练的模型
py inference.py --model_path models/best_real_data_model.pth --input data/ --visualize

# 对单个文件预测
py inference.py --model_path models/best_real_data_model.pth --input your_audio.wav

# 生成详细报告
py inference.py --model_path models/best_real_data_model.pth --input data/ --output results.json --visualize --viz_output analysis.png
```

## 🎵 支持的声音类型

| 类型 | 中文名 | 情绪 | 商业含义 |
|-----|-------|-----|---------|
| `laughing` | 笑声 | 积极 | 顾客满意的正向反馈 |
| `sighing` | 叹气 | 消极 | 等待过久或不满情绪 |
| `tongue_clicking` | 咂舌 | 消极 | 焦躁或不耐烦 |
| `throat_clearing` | 清嗓 | 中性 | 紧张或试图引起注意 |
| `teeth_grinding` | 磨牙 | 消极 | 强烈挫败或愤怒 |
| `yawning` | 打哈欠 | 中性 | 厌烦或无聊 |
| `lip_smacking` | 抿嘴 | 积极 | 期待或轻度愉悦 |

## ⚙️ 快速配置

### 根据数据量调整设置

**小数据集 (<50样本/类别)**:
```bash
py train_real_data.py --batch_size 4 --epochs 50 --learning_rate 5e-5
```

**中等数据集 (50-200样本/类别)**:
```bash
py train_real_data.py --batch_size 16 --epochs 30 --learning_rate 1e-4
```

**大数据集 (>200样本/类别)**:
```bash
py train_real_data.py --batch_size 32 --epochs 50 --learning_rate 1e-4 --use_wandb
```

### 模型选择

```bash
# AST模型（推荐，但需要更多内存）
py train_real_data.py --model_type ast

# CNN模型（较快，内存占用少）
py train_real_data.py --model_type cnn
```

## 🔍 查看结果

### 训练完成后检查
- `models/best_real_data_model.pth` - 最佳模型
- `logs/real_data_confusion_matrix.png` - 混淆矩阵
- `logs/real_data_training_history.png` - 训练曲线
- `logs/real_data_test_results.json` - 详细测试结果

### 推理完成后查看
- `results.json` - 预测结果
- `analysis.png` - 可视化图表

## 🆘 常见问题

**Q: 报错 "No module named 'torch'"**
A: 运行 `pip install -r requirements.txt`

**Q: 目录名称不匹配**
A: 运行 `py organize_data.py` 自动修复

**Q: CUDA内存不足**
A: 使用 `--batch_size 4` 或 `--model_type cnn`

**Q: 训练很慢**
A: 
- 使用GPU：确保安装了CUDA版本的PyTorch
- 减少数据：删除部分训练样本
- 使用CNN模型：`--model_type cnn`

**Q: 准确率很低**
A: 
- 检查数据质量和标注正确性
- 增加训练数据
- 调整训练参数：`--epochs 50 --learning_rate 5e-5`

**Q: 数据不平衡**
A: 
```bash
# 检查数据分布
py organize_data.py --analyze_only

# 为少数类别生成合成数据
py data_generator.py
```

## 📈 性能预期

基于313个真实音频文件的测试：

- **训练时间**: 10-30分钟（取决于硬件）
- **预期准确率**: 80-95%（取决于数据质量）
- **内存需求**: 2-8GB（AST需要更多）
- **最小数据量**: 每类别至少5个样本

## 📞 获取帮助

- 查看完整文档：`README.md`
- 查看命令帮助：`py train_real_data.py --help`
- 数据分析：`py organize_data.py --analyze_only`
- 项目结构：`py demo.py --info-only`

---

🎉 现在你已经可以开始使用真实数据训练声音分类系统了！ 
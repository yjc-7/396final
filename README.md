# 声音分类系统 (Sound Classification System)

基于 Audio Spectrogram Transformer (AST) 的顾客行为声音分类系统，用于识别和分析顾客在商业环境中的情绪和行为状态。

## 🎯 项目概述

本项目旨在通过分析声音信号来识别顾客的情绪状态和行为模式，帮助商家更好地理解顾客体验。系统能够识别以下7种声音类型：

| 声音类型 | 中文名称 | 情绪类别 | 商业含义 |
|---------|---------|---------|---------|
| `laughing` | 笑声 | 积极 | 顾客满意度高，体验良好 |
| `sighing` | 叹气 | 消极 | 等待过久或不满情绪 |
| `tongue_clicking` | 咂舌 | 消极 | 对服务不满，表达焦躁 |
| `throat_clearing` | 清嗓 | 中性 | 等待中的紧张或尝试引起注意 |
| `teeth_grinding` | 磨牙 | 消极 | 强烈挫败感或愤怒 |
| `yawning` | 打哈欠 | 中性 | 厌烦或对体验不感兴趣 |
| `lip_smacking` | 抿嘴/咂嘴 | 积极 | 期待或轻度愉悦 |

## 🏗️ 系统架构

### 核心技术
- **模型**: Audio Spectrogram Transformer (AST) - 当前音频分类的 SOTA 模型
- **备选模型**: 自定义 CNN 模型（用于资源受限环境）
- **特征提取**: Mel频谱图 + 音频增强
- **框架**: PyTorch + Transformers

### 项目结构
```
FINAL/
├── config.py              # 配置文件和参数设置
├── data_preprocessing.py   # 数据预处理和增强
├── model.py               # 模型定义（AST + CNN）
├── train.py               # 基础训练脚本
├── train_real_data.py     # 🆕 真实数据专用训练脚本
├── inference.py           # 推理脚本
├── data_generator.py      # 合成数据生成器
├── organize_data.py       # 🆕 数据组织和分析工具
├── requirements.txt       # 依赖包列表
├── README.md             # 项目说明文档
├── data/                 # 数据目录
│   ├── laughing/         # 笑声样本
│   ├── sighing/          # 叹气样本
│   ├── tongue_clicking/  # 咂舌样本
│   ├── throat_clearing/  # 清嗓样本
│   ├── teeth_grinding/   # 磨牙样本
│   ├── yawning/          # 打哈欠样本
│   └── lip_smacking/     # 抿嘴样本
├── models/               # 保存的模型
└── logs/                # 训练日志和结果
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用conda
conda create -n sound_classification python=3.8
conda activate sound_classification
pip install -r requirements.txt
```

### 2. 数据准备

#### 选项A: 使用真实数据（推荐）
将音频文件按类别放入相应目录：
```
data/
├── laughing/          # 或 laughing-sounds/ 等变体
│   ├── real_laugh_001.wav
│   └── real_laugh_002.wav
├── sighing/           # 或 sighing-sounds/ 等变体
│   ├── real_sigh_001.wav
│   └── real_sigh_002.wav
└── ...
```

**支持的音频格式**: WAV, MP3, FLAC, M4A, OGG
**目录命名**: 支持下划线、连字符等变体（如 `tongue-clicking` 或 `tongue_clicking`）

#### 选项B: 自动数据组织
如果你的数据目录使用不同的命名格式：

```bash
# 自动重命名目录并分析数据
py organize_data.py

# 仅分析数据分布
py organize_data.py --analyze_only

# 仅清理无效文件
py organize_data.py --clean_only
```

#### 选项C: 使用合成数据（测试用）
```bash
# 生成合成训练数据
py data_generator.py
```

### 3. 模型训练

#### 真实数据训练（推荐）
```bash
# 使用真实数据训练（默认30个epoch）
py train_real_data.py

# 自定义参数
py train_real_data.py --epochs 50 --batch_size 16 --learning_rate 1e-4

# 使用Weights & Biases记录
py train_real_data.py --use_wandb

# 使用CNN模型（如果AST不可用）
py train_real_data.py --model_type cnn
```

#### 基础训练
```bash
# 传统训练方式
py train.py
```

### 4. 模型推理

```bash
# 使用真实数据训练的模型
py inference.py --model_path models/best_real_data_model.pth --input audio_file.wav

# 对目录中所有文件批量预测
py inference.py --model_path models/best_real_data_model.pth --input audio_directory/ --visualize

# 生成可视化结果
py inference.py --model_path models/best_real_data_model.pth --input data/ --output results.json --visualize --viz_output visualization.png
```

## 📊 数据集信息

### 支持的数据格式
- **文件格式**: WAV, MP3, FLAC, M4A, OGG
- **采样率**: 自动转换为16kHz
- **时长**: 自动调整为5秒（截取或填充）
- **通道**: 自动转换为单声道

### 目录结构灵活性
系统自动识别多种目录命名格式：
- 标准格式: `tongue_clicking`, `throat_clearing`
- 连字符格式: `tongue-clicking`, `throat-clearing`  
- 其他变体: `tongueclicking`, `tongue clicking`

### 数据组织工具
```bash
# 完整数据组织流程
py organize_data.py

# 分析当前数据状态
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

## 🎓 训练配置

### 真实数据训练优化
`train_real_data.py` 专门为真实数据优化：

- **学习率调度**: ReduceLROnPlateau（更保守）
- **早停策略**: 15个epoch耐心（更宽容）
- **优化器**: 对AST预训练层使用更低学习率
- **数据增强**: 适合真实数据的增强策略
- **错误处理**: 更好的批处理错误恢复

### 默认配置
- **模型**: MIT/ast-finetuned-audioset-10-10-0.4593
- **音频参数**: 16kHz采样率，5秒时长，128个Mel频带
- **训练参数**: 
  - 批大小: 16
  - 学习率: 1e-4（AST层：1e-5）
  - 优化器: AdamW
  - 调度器: ReduceLROnPlateau
  - 早停: 15个epoch

### 数据增强
- 高斯噪声添加
- 时间拉伸 (0.8x - 1.25x)
- 音调偏移 (±4半音)
- 时间偏移

## 🛠️ 高级使用

### 自定义配置

编辑 `config.py` 中的参数：

```python
class Config:
    # 音频参数
    sample_rate: int = 16000
    max_audio_length: int = 5
    
    # 模型参数  
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    num_classes: int = 7
    
    # 训练参数
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 30  # 真实数据推荐30-50个epoch
```

### 添加新的声音类别

1. 在 `config.py` 中添加新类别名称
2. 在 `SOUND_DESCRIPTIONS` 和 `EMOTION_MAPPING` 中添加描述
3. 创建对应的数据目录并添加音频样本
4. 运行 `py organize_data.py` 检查数据
5. 重新训练模型

### 使用不同的模型

```python
# 在 train_real_data.py 中切换模型
py train_real_data.py --model_type cnn   # 使用CNN模型
py train_real_data.py --model_type ast   # 使用AST模型（默认）
```

## 📈 监控和可视化

### 训练监控
- 训练/验证损失和准确率曲线
- 学习率变化
- 混淆矩阵（带百分比）
- 分类报告
- 真实数据vs合成数据统计

### 推理结果可视化
- 预测类别分布
- 置信度分布  
- 情绪分类饼图
- 各类别置信度箱线图

### 输出文件
训练完成后检查：
- `models/best_real_data_model.pth` - 最佳模型
- `logs/real_data_confusion_matrix.png` - 混淆矩阵
- `logs/real_data_training_history.png` - 训练曲线
- `logs/real_data_test_results.json` - 详细测试结果

## 🔧 故障排除

### 常见问题

1. **数据目录名称不匹配**
   ```bash
   # 自动修复目录命名
   py organize_data.py
   ```

2. **CUDA内存不足**
   ```python
   # 减少批大小
   py train_real_data.py --batch_size 8
   
   # 或使用CPU
   device = torch.device("cpu")
   ```

3. **音频文件加载失败**
   ```bash
   # 清理无效文件
   py organize_data.py --clean_only
   ```

4. **模型下载失败**
   ```bash
   # 使用CNN模型
   py train_real_data.py --model_type cnn
   ```

5. **数据不平衡**
   ```bash
   # 检查数据分布
   py organize_data.py --analyze_only
   
   # 为缺少的类别生成合成数据
   py data_generator.py --classes tongue_clicking lip_smacking
   ```

### 性能优化建议

根据数据集大小选择配置：

- **小数据集 (<100样本/类别)**:
  ```bash
  py train_real_data.py --batch_size 4 --epochs 50 --learning_rate 5e-5
  ```

- **中等数据集 (100-500样本/类别)**:
  ```bash
  py train_real_data.py --batch_size 16 --epochs 30 --learning_rate 1e-4
  ```

- **大数据集 (>500样本/类别)**:
  ```bash
  py train_real_data.py --batch_size 32 --epochs 50 --learning_rate 1e-4 --use_wandb
  ```

## 📝 API参考

### SoundClassifier类

```python
from inference import SoundClassifier

# 初始化分类器（使用真实数据训练的模型）
classifier = SoundClassifier("models/best_real_data_model.pth")

# 单文件预测
result = classifier.predict_single("audio.wav")
print(f"类别: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.3f}")

# 批量预测
results = classifier.predict_batch(["audio1.wav", "audio2.wav"])

# 目录预测
results = classifier.predict_directory("audio_directory/")
```

### 数据组织工具

```python
from organize_data import analyze_data_distribution

# 分析数据分布
analyze_data_distribution("data/")

# 自动重命名目录
from organize_data import rename_directories
rename_directories("data/")
```

### 输出格式

```json
{
  "audio_path": "path/to/audio.wav",
  "predicted_class": "laughing",
  "confidence": 0.924,
  "emotion": "positive", 
  "description": "顾客满意或享受体验时常会发出的正向反馈",
  "all_probabilities": {
    "laughing": 0.924,
    "sighing": 0.041,
    "tongue_clicking": 0.018,
    "throat_clearing": 0.012,
    "teeth_grinding": 0.003,
    "yawning": 0.001,
    "lip_smacking": 0.001
  }
}
```

## 📊 实际数据结果

基于真实音频数据的测试结果：

- **数据集规模**: 313个真实音频文件
- **类别平衡度**: 2.2（优秀）
- **建议训练配置**: 中等设置（batch_size=16, epochs=30-50）
- **预期准确率**: 80-95%（取决于数据质量）

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Audio Spectrogram Transformer](https://github.com/YuanGongND/ast) - 核心模型架构
- [Hugging Face Transformers](https://huggingface.co/transformers/) - 预训练模型
- [librosa](https://librosa.org/) - 音频处理库

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 创建 [Issue](https://github.com/your-repo/issues)
- 发送邮件到: your-email@example.com

---

⭐ 如果这个项目对您有帮助，请给个星标！ 
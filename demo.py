#!/usr/bin/env python3
"""
声音分类系统演示脚本
演示完整的数据生成、训练和推理流程
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def print_header(text):
    """打印标题"""
    print("\n" + "="*60)
    print(f" {text}")
    print("="*60)

def print_step(step_num, text):
    """打印步骤"""
    print(f"\n🚀 步骤 {step_num}: {text}")
    print("-" * 40)

def run_command(command, description):
    """运行命令并显示结果"""
    print(f"执行: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} 成功完成")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print(f"❌ {description} 失败")
            print("错误信息:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False
    return True

def check_dependencies():
    """检查依赖是否安装"""
    print_step(0, "检查环境和依赖")
    
    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print("❌ 需要Python 3.7或更高版本")
        return False
    
    # 检查关键依赖
    required_packages = ['torch', 'librosa', 'transformers', 'soundfile', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n请先安装缺失的依赖:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def generate_demo_data():
    """生成演示数据"""
    print_step(1, "生成演示数据")
    
    # 检查是否已有数据
    data_dir = Path("data")
    if data_dir.exists() and any(data_dir.iterdir()):
        print("📁 检测到已有数据目录")
        response = input("是否重新生成数据? (y/n): ").lower().strip()
        if response != 'y':
            print("跳过数据生成步骤")
            return True
    
    print("🎵 开始生成合成音频数据...")
    
    # 生成数据
    command = "python data_generator.py"
    return run_command(command, "数据生成")

def quick_train():
    """快速训练演示"""
    print_step(2, "快速模型训练")
    
    print("⚠️  注意: 这是演示训练，仅运行少量epoch")
    print("完整训练请直接运行: python train.py")
    
    # 创建快速训练配置
    quick_config = """
# 快速训练配置
import sys
sys.path.append('.')
from config import Config
from train import Trainer

# 修改配置以进行快速训练
config = Config()
config.num_epochs = 3  # 只训练3个epoch用于演示
config.batch_size = 8   # 减小批大小
config.patience = 5     # 减少早停耐心

print("开始快速训练演示...")
trainer = Trainer(config, use_wandb=False)

try:
    history = trainer.train()
    print("✅ 快速训练完成!")
    
    # 显示简单结果
    if history['val_accuracy']:
        final_acc = history['val_accuracy'][-1]
        print(f"最终验证准确率: {final_acc:.3f}")
        
except Exception as e:
    print(f"❌ 训练过程中出错: {e}")
    print("这可能是由于数据不足或配置问题导致的")
"""
    
    # 写入临时训练脚本
    with open("quick_train.py", "w", encoding="utf-8") as f:
        f.write(quick_config)
    
    # 运行快速训练
    command = "python quick_train.py"
    success = run_command(command, "快速训练")
    
    # 清理临时文件
    if os.path.exists("quick_train.py"):
        os.remove("quick_train.py")
    
    return success

def demo_inference():
    """演示推理功能"""
    print_step(3, "演示模型推理")
    
    # 检查是否有训练好的模型
    model_path = "models/best_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ 未找到训练好的模型: {model_path}")
        print("请先运行训练步骤")
        return False
    
    print("🔍 使用训练好的模型进行推理演示...")
    
    # 对数据目录进行推理
    command = f"python inference.py --model_path {model_path} --input data/ --output demo_results.json --visualize --viz_output demo_visualization.png"
    
    success = run_command(command, "推理演示")
    
    if success:
        print("\n📊 推理结果已保存:")
        print("- demo_results.json: 详细预测结果")
        print("- demo_visualization.png: 可视化图表")
    
    return success

def show_project_structure():
    """显示项目结构"""
    print_step("info", "项目结构概览")
    
    structure = """
    项目文件说明:
    
    📁 核心文件:
    ├── config.py              # 配置文件 - 修改模型和训练参数
    ├── data_preprocessing.py   # 数据处理 - 音频加载和增强
    ├── model.py               # 模型定义 - AST和CNN模型
    ├── train.py               # 训练脚本 - 完整训练流程
    ├── inference.py           # 推理脚本 - 模型预测
    └── data_generator.py      # 数据生成 - 合成训练数据
    
    📁 数据目录:
    ├── data/                  # 训练数据
    │   ├── laughing/          # 笑声样本
    │   ├── sighing/           # 叹气样本
    │   ├── tongue_clicking/   # 咂舌样本
    │   ├── throat_clearing/   # 清嗓样本
    │   ├── teeth_grinding/    # 磨牙样本
    │   ├── yawning/           # 打哈欠样本
    │   └── lip_smacking/      # 抿嘴样本
    
    📁 输出目录:
    ├── models/                # 保存的模型文件
    └── logs/                  # 训练日志和结果
    """
    
    print(structure)

def main():
    parser = argparse.ArgumentParser(description="声音分类系统演示")
    parser.add_argument('--skip-deps', action='store_true', help='跳过依赖检查')
    parser.add_argument('--skip-data', action='store_true', help='跳过数据生成')
    parser.add_argument('--skip-train', action='store_true', help='跳过训练演示')
    parser.add_argument('--skip-inference', action='store_true', help='跳过推理演示')
    parser.add_argument('--info-only', action='store_true', help='仅显示项目信息')
    
    args = parser.parse_args()
    
    print_header("🎵 声音分类系统演示")
    print("基于 Audio Spectrogram Transformer 的顾客行为声音识别系统")
    
    if args.info_only:
        show_project_structure()
        return
    
    # 检查依赖
    if not args.skip_deps:
        if not check_dependencies():
            print("\n❌ 环境检查失败，请先安装依赖")
            print("运行: pip install -r requirements.txt")
            return
    
    # 生成演示数据
    if not args.skip_data:
        if not generate_demo_data():
            print("\n❌ 数据生成失败")
            return
    
    # 快速训练演示
    if not args.skip_train:
        if not quick_train():
            print("\n⚠️  训练演示失败，但可以继续其他步骤")
            print("对于完整训练，请运行: python train.py")
    
    # 推理演示
    if not args.skip_inference:
        if not demo_inference():
            print("\n⚠️  推理演示失败")
            print("请确保有训练好的模型文件")
    
    # 显示完成信息
    print_header("🎉 演示完成")
    
    completion_info = """
    演示已完成! 接下来你可以:
    
    🔧 自定义配置:
    - 编辑 config.py 调整模型参数
    - 添加自己的音频数据到对应目录
    
    🚀 完整训练:
    - 运行 python train.py 进行完整训练
    - 使用 --use_wandb=True 启用训练监控
    
    🔍 模型推理:
    - python inference.py --model_path models/best_model.pth --input your_audio.wav
    - python inference.py --help 查看更多选项
    
    📚 查看文档:
    - 阅读 README.md 了解详细使用说明
    - 检查生成的结果文件和可视化图表
    
    ⭐ 项目地址: https://github.com/your-repo/sound-classification
    """
    
    print(completion_info)

if __name__ == "__main__":
    main() 
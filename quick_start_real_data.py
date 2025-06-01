#!/usr/bin/env python3
"""
真实数据快速开始脚本
一键运行数据组织、训练和推理的完整流程
"""

import os
import subprocess
import sys
from pathlib import Path

def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_step(step, description):
    """打印步骤"""
    print(f"\n🚀 步骤 {step}: {description}")
    print("-" * 40)

def run_command(command, description, show_output=True):
    """运行命令"""
    print(f"执行: {command}")
    try:
        if show_output:
            result = subprocess.run(command, shell=True, text=True)
            success = result.returncode == 0
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            success = result.returncode == 0
            if result.stdout:
                print(result.stdout)
            if result.stderr and not success:
                print(f"错误: {result.stderr}")
        
        if success:
            print(f"✅ {description} 完成")
        else:
            print(f"❌ {description} 失败")
        
        return success
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False

def check_data_exists():
    """检查是否有数据"""
    data_dir = Path("data")
    if not data_dir.exists():
        return False
    
    # 检查是否有音频文件
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            for file in subdir.iterdir():
                if any(file.name.lower().endswith(ext) for ext in audio_extensions):
                    return True
    return False

def main():
    print_header("🎵 真实数据声音分类系统 - 快速开始")
    
    # 检查Python命令
    python_cmd = "py" if os.name == 'nt' else "python"
    
    # 步骤1: 检查数据
    print_step(1, "检查数据")
    if not check_data_exists():
        print("❌ 未找到音频数据文件")
        print("请将音频文件放入data目录的对应子目录中")
        print("支持的格式: WAV, MP3, FLAC, M4A, OGG")
        return
    else:
        print("✅ 找到音频数据文件")
    
    # 步骤2: 组织数据
    print_step(2, "组织和分析数据")
    success = run_command(f"{python_cmd} organize_data.py", "数据组织")
    if not success:
        print("⚠️ 数据组织失败，但可以继续训练")
    
    # 步骤3: 训练模型
    print_step(3, "训练模型")
    print("开始使用真实数据训练模型...")
    print("这可能需要几分钟到几小时，取决于数据量和硬件配置")
    
    # 使用适中的参数进行训练
    train_cmd = f"{python_cmd} train_real_data.py --epochs 20 --batch_size 16"
    success = run_command(train_cmd, "模型训练", show_output=True)
    
    if not success:
        print("⚠️ 训练失败，尝试使用CNN模型...")
        train_cmd_fallback = f"{python_cmd} train_real_data.py --model_type cnn --epochs 20 --batch_size 8"
        success = run_command(train_cmd_fallback, "CNN模型训练", show_output=True)
    
    if not success:
        print("❌ 训练失败，请检查错误信息")
        return
    
    # 步骤4: 测试推理
    print_step(4, "测试推理")
    
    # 检查是否有训练好的模型
    model_path = "models/best_real_data_model.pth"
    if os.path.exists(model_path):
        print(f"使用模型: {model_path}")
        inference_cmd = f"{python_cmd} inference.py --model_path {model_path} --input data/ --output real_data_results.json --visualize --viz_output real_data_visualization.png"
        success = run_command(inference_cmd, "批量推理测试")
        
        if success:
            print("\n📊 推理结果已保存:")
            print("- real_data_results.json: 详细预测结果")
            print("- real_data_visualization.png: 可视化图表")
    else:
        print(f"❌ 未找到训练好的模型: {model_path}")
    
    # 完成总结
    print_header("🎉 快速开始完成")
    
    completion_info = """
    恭喜！你已经成功完成了真实数据的声音分类系统搭建！
    
    📁 检查输出文件:
    - models/best_real_data_model.pth - 训练好的模型
    - logs/real_data_confusion_matrix.png - 混淆矩阵
    - logs/real_data_training_history.png - 训练曲线
    - real_data_results.json - 推理结果
    - real_data_visualization.png - 结果可视化
    
    🔍 下一步操作:
    1. 查看训练结果和性能指标
    2. 使用模型对新音频进行预测:
       py inference.py --model_path models/best_real_data_model.pth --input your_audio.wav
    
    3. 如需改进模型:
       - 添加更多训练数据
       - 调整训练参数: py train_real_data.py --epochs 50 --batch_size 16
       - 尝试数据增强和预处理优化
    
    📚 更多信息:
    - 阅读 README.md 了解详细使用说明
    - 查看 logs/ 目录中的训练日志和结果分析
    """
    
    print(completion_info)

if __name__ == "__main__":
    main() 
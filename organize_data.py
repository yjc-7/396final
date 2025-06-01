#!/usr/bin/env python3
"""
数据组织脚本
自动重命名数据目录并整理真实数据，使其与代码配置兼容
"""

import os
import shutil
from pathlib import Path
import argparse

from config import Config

def rename_directories(data_dir):
    """重命名数据目录以匹配配置"""
    config = Config()
    
    # 目录映射：从实际目录名到配置中的标准名
    directory_mapping = {
        'laughing': 'laughing',
        'sighing': 'sighing',
        'tongue-clicking': 'tongue_clicking',
        'throat-clearing': 'throat_clearing',
        'teeth-grinding': 'teeth_grinding',
        'yawning': 'yawning',
        'lip-smacking': 'lip_smacking'
    }
    
    print("🔄 Organizing data directories...")
    
    for actual_name, standard_name in directory_mapping.items():
        actual_path = os.path.join(data_dir, actual_name)
        standard_path = os.path.join(data_dir, standard_name)
        
        if os.path.exists(actual_path) and actual_name != standard_name:
            print(f"  Renaming: {actual_name} -> {standard_name}")
            
            # 如果目标目录已存在，合并文件
            if os.path.exists(standard_path):
                print(f"  Merging files into existing {standard_name} directory")
                merge_directories(actual_path, standard_path)
                shutil.rmtree(actual_path)
            else:
                # 直接重命名
                shutil.move(actual_path, standard_path)
        elif os.path.exists(standard_path):
            print(f"  ✓ {standard_name} directory already exists")
        else:
            print(f"  ⚠️  No directory found for {actual_name} or {standard_name}")

def merge_directories(src_dir, dst_dir):
    """合并两个目录的文件"""
    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dst_path = os.path.join(dst_dir, item)
        
        if os.path.isfile(src_path):
            # 如果目标文件已存在，重命名
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(item)
                counter = 1
                while os.path.exists(dst_path):
                    new_name = f"{name}_copy{counter}{ext}"
                    dst_path = os.path.join(dst_dir, new_name)
                    counter += 1
            
            shutil.move(src_path, dst_path)

def clean_data_files(data_dir):
    """清理数据文件，移除无效文件"""
    config = Config()
    
    print("\n🧹 Cleaning data files...")
    
    total_removed = 0
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    for class_name in config.class_names:
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            continue
        
        removed_count = 0
        files_to_remove = []
        
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            
            # 移除非音频文件
            if not any(file_name.lower().endswith(ext) for ext in audio_extensions):
                if not file_name.startswith('.'):  # 保留隐藏文件
                    files_to_remove.append(file_path)
                    print(f"    Removing non-audio file: {file_name}")
                continue
            
            # 检查文件大小
            try:
                if os.path.getsize(file_path) == 0:
                    files_to_remove.append(file_path)
                    print(f"    Removing empty file: {file_name}")
                    continue
            except OSError:
                files_to_remove.append(file_path)
                print(f"    Removing inaccessible file: {file_name}")
                continue
        
        # 删除标记的文件
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                removed_count += 1
            except OSError as e:
                print(f"    Failed to remove {file_path}: {e}")
        
        if removed_count > 0:
            print(f"  Cleaned {removed_count} files from {class_name}")
        
        total_removed += removed_count
    
    print(f"  Total files removed: {total_removed}")

def analyze_data_distribution(data_dir):
    """分析数据分布"""
    config = Config()
    
    print("\n📊 Data distribution analysis:")
    
    total_files = 0
    class_counts = {}
    real_data_files = []
    synthetic_data_files = []
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    for class_name in config.class_names:
        class_dir = os.path.join(data_dir, class_name)
        
        if not os.path.exists(class_dir):
            class_counts[class_name] = 0
            continue
        
        count = 0
        for file_name in os.listdir(class_dir):
            if any(file_name.lower().endswith(ext) for ext in audio_extensions):
                file_path = os.path.join(class_dir, file_name)
                try:
                    if os.path.getsize(file_path) > 0:
                        count += 1
                        total_files += 1
                        
                        # 判断是真实数据还是合成数据
                        if file_name.startswith(f"{class_name}_") and file_name.endswith('.wav'):
                            # 检查是否为合成数据的命名模式
                            try:
                                # 提取文件名中的数字部分
                                number_part = file_name[len(class_name)+1:-4]  # 去掉前缀和.wav
                                if number_part.isdigit() and len(number_part) == 3:
                                    synthetic_data_files.append(file_path)
                                else:
                                    real_data_files.append(file_path)
                            except:
                                real_data_files.append(file_path)
                        else:
                            real_data_files.append(file_path)
                except OSError:
                    continue
        
        class_counts[class_name] = count
    
    print(f"  Total files: {total_files}")
    print(f"  Real data files: {len(real_data_files)}")
    print(f"  Synthetic data files: {len(synthetic_data_files)}")
    
    print(f"\n  Class distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / total_files * 100) if total_files > 0 else 0
        print(f"    {class_name}: {count} files ({percentage:.1f}%)")
    
    # 检查数据平衡
    if class_counts:
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        
        if max_count > 0 and min_count > 0:
            imbalance_ratio = max_count / min_count
            if imbalance_ratio > 3:
                print(f"\n  ⚠️  Dataset is imbalanced! Ratio: {imbalance_ratio:.1f}")
                print(f"    Most samples: {max_count}, Least samples: {min_count}")
                print("    Consider generating synthetic data for underrepresented classes.")
            else:
                print(f"\n  ✓ Dataset is reasonably balanced (ratio: {imbalance_ratio:.1f})")
        
        if min_count < 5:
            print(f"\n  ⚠️  Some classes have very few samples (minimum: {min_count})")
            print("    Consider adding more data or generating synthetic samples.")

def suggest_improvements(data_dir):
    """建议数据改进方案"""
    config = Config()
    
    print("\n💡 Suggestions for improvement:")
    
    class_counts = {}
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    for class_name in config.class_names:
        class_dir = os.path.join(data_dir, class_name)
        count = 0
        
        if os.path.exists(class_dir):
            for file_name in os.listdir(class_dir):
                if any(file_name.lower().endswith(ext) for ext in audio_extensions):
                    try:
                        file_path = os.path.join(class_dir, file_name)
                        if os.path.getsize(file_path) > 0:
                            count += 1
                    except OSError:
                        continue
        
        class_counts[class_name] = count
    
    # 找出样本数量少的类别
    if class_counts:
        avg_count = sum(class_counts.values()) / len(class_counts)
        low_sample_classes = [name for name, count in class_counts.items() if count < avg_count * 0.5]
        
        if low_sample_classes:
            print(f"  Classes needing more data: {', '.join(low_sample_classes)}")
            print(f"  Recommendation: Generate synthetic data using:")
            print(f"    python data_generator.py --classes {' '.join(low_sample_classes)}")
        
        # 推荐训练配置
        min_samples = min(class_counts.values()) if class_counts.values() else 0
        total_samples = sum(class_counts.values())
        
        if total_samples > 0:
            if min_samples < 10:
                print(f"  Training recommendation: Use smaller batch size (4-8)")
                print(f"  Training recommendation: Increase data augmentation strength")
            elif total_samples > 500:
                print(f"  Training recommendation: Use larger batch size (16-32)")
                print(f"  Training recommendation: Consider longer training (100+ epochs)")
            else:
                print(f"  Training recommendation: Use moderate settings (batch_size=16, epochs=50)")

def main():
    parser = argparse.ArgumentParser(description="组织和分析音频数据")
    parser.add_argument('--data_dir', type=str, default='data', help='数据目录路径')
    parser.add_argument('--rename_only', action='store_true', help='仅重命名目录')
    parser.add_argument('--clean_only', action='store_true', help='仅清理文件')
    parser.add_argument('--analyze_only', action='store_true', help='仅分析数据')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' does not exist!")
        return
    
    print(f"📁 Processing data directory: {data_dir}")
    
    if args.analyze_only:
        analyze_data_distribution(data_dir)
        suggest_improvements(data_dir)
    elif args.clean_only:
        clean_data_files(data_dir)
        analyze_data_distribution(data_dir)
    elif args.rename_only:
        rename_directories(data_dir)
        analyze_data_distribution(data_dir)
    else:
        # 完整流程
        rename_directories(data_dir)
        clean_data_files(data_dir)
        analyze_data_distribution(data_dir)
        suggest_improvements(data_dir)
    
    print("\n✅ Data organization completed!")

if __name__ == "__main__":
    main() 
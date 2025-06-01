#!/usr/bin/env python3
"""
环境检查脚本
验证所有依赖包是否正确安装
"""

import sys
import importlib

def check_package(package_name, min_version=None, import_name=None):
    """检查包是否安装且版本符合要求"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        
        # 获取版本号
        pkg_version = "unknown"
        if hasattr(module, '__version__'):
            pkg_version = module.__version__
        elif hasattr(module, 'version'):
            pkg_version = module.version
        elif hasattr(module, '__version'):
            pkg_version = module.__version
        
        # 检查版本
        version_ok = True
        if min_version and pkg_version != "unknown":
            try:
                from packaging import version
                version_ok = version.parse(pkg_version) >= version.parse(min_version)
            except:
                version_ok = True  # 无法解析版本时认为OK
        
        status = "✅" if version_ok else "⚠️"
        print(f"{status} {package_name}: {pkg_version}")
        
        if min_version and not version_ok:
            print(f"   Warning: Requires >= {min_version}")
        
        return True, version_ok
        
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False, False

def main():
    print("🔍 Checking environment dependencies...")
    print("=" * 50)
    
    # Python版本检查
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python: {python_version}")
    
    if sys.version_info < (3, 7):
        print("⚠️  Warning: Python 3.7+ is recommended")
    
    print("\n📦 Package versions:")
    print("-" * 30)
    
    # 核心依赖 - 更新版本要求
    packages = [
        ("torch", "1.11.0"),
        ("torchaudio", "0.11.0"),
        ("transformers", "4.20.0"),
        ("librosa", "0.9.0"),
        ("soundfile", "0.10.0"),
        ("numpy", "1.21.0"),
        ("scikit-learn", "1.0.0", "sklearn"),
        ("matplotlib", "3.5.0"),
        ("seaborn", "0.11.0"),
        ("pandas", "1.3.0"),
        ("tqdm", "4.60.0"),
    ]
    
    # 可选依赖
    optional_packages = [
        ("jupyter", None),
        ("tensorboard", "2.8.0"),
        ("wandb", "0.13.0"),
        ("audiomentations", "0.30.0"),
        ("packaging", None),
    ]
    
    all_installed = True
    version_issues = []
    
    # 检查核心依赖
    for pkg_info in packages:
        if len(pkg_info) == 3:
            pkg_name, min_ver, import_name = pkg_info
        else:
            pkg_name, min_ver = pkg_info
            import_name = None
        
        installed, version_ok = check_package(pkg_name, min_ver, import_name)
        if not installed:
            all_installed = False
        if not version_ok:
            version_issues.append(pkg_name)
    
    print("\n📦 Optional packages:")
    print("-" * 30)
    
    # 检查可选依赖
    for pkg_info in optional_packages:
        if len(pkg_info) == 3:
            pkg_name, min_ver, import_name = pkg_info
        else:
            pkg_name, min_ver = pkg_info
            import_name = None
        
        check_package(pkg_name, min_ver, import_name)
    
    # 设备检查
    print("\n🖥️  Device information:")
    print("-" * 30)
    
    try:
        import torch
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
        else:
            print("Running on CPU (consider installing CUDA for faster training)")
    except ImportError:
        print("❌ PyTorch not available for device check")
    
    # 总结
    print("\n" + "=" * 50)
    if all_installed and not version_issues:
        print("🎉 All dependencies are correctly installed!")
        print("You can now run the sound classification system.")
    elif all_installed:
        print("⚠️  All packages installed but some version warnings.")
        print("The system should still work, but consider updating packages.")
    else:
        print("❌ Some required packages are missing.")
        print("Please install missing packages before proceeding.")
    
    # 安装建议
    if not all_installed or version_issues:
        print("\n💡 Installation commands:")
        print("For conda environment:")
        print("  conda env create -f environment.yml")
        print("For pip installation:")
        print("  pip install -r requirements.txt")
    
    # 快速测试
    print("\n🚀 Quick functionality test:")
    print("-" * 30)
    
    try:
        import torch
        import librosa
        import numpy as np
        
        # 测试音频处理
        dummy_audio = np.random.randn(16000)  # 1秒的随机音频
        mel_spec = librosa.feature.melspectrogram(y=dummy_audio, sr=16000)
        print("✅ Audio processing test passed")
        
        # 测试PyTorch tensor操作
        tensor = torch.tensor(mel_spec)
        print("✅ PyTorch tensor test passed")
        
        print("✅ Basic functionality test completed successfully!")
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")

if __name__ == "__main__":
    main() 
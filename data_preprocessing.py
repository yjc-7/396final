import os
import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import warnings
warnings.filterwarnings("ignore")

from config import Config

class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, config, transform=None, is_training=True):
        self.audio_paths = audio_paths
        self.labels = labels
        self.config = config
        self.transform = transform
        self.is_training = is_training
        
        # 音频增强
        if is_training:
            self.augment = Compose([
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.3),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.3),
                Shift(min_shift=-0.5, max_shift=0.5, p=0.3),
            ])
        else:
            self.augment = None
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # 加载音频
        audio = self.load_audio(audio_path)
        
        # 应用增强（仅训练时）
        if self.is_training and self.augment:
            audio = self.augment(audio, sample_rate=self.config.sample_rate)
        
        # 转换为tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        
        return {
            'audio': audio_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'path': audio_path
        }
    
    def load_audio(self, audio_path):
        """加载和预处理音频文件"""
        try:
            # 使用librosa加载音频
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            # 规范化音频长度
            audio = self.normalize_audio_length(audio)
            
            # 归一化音频幅度
            audio = librosa.util.normalize(audio)
            
            return audio
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # 返回静音
            return np.zeros(self.config.sample_rate * self.config.max_audio_length)
    
    def normalize_audio_length(self, audio):
        """规范化音频长度"""
        target_length = self.config.sample_rate * self.config.max_audio_length
        
        if len(audio) > target_length:
            # 如果音频过长，截取中间部分
            start = (len(audio) - target_length) // 2
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            # 如果音频过短，用零填充
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
        
        return audio

def extract_mel_spectrogram(audio, config):
    """提取Mel频谱图特征"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=config.sample_rate,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        hop_length=config.hop_length
    )
    
    # 转换为对数尺度
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return log_mel_spec

def get_directory_variants(class_name):
    """获取类别目录名的所有可能变体"""
    variants = [
        class_name,  # 原始名称 (tongue_clicking)
        class_name.replace('_', '-'),  # 连字符版本 (tongue-clicking)
        class_name.replace('_', ''),   # 无分隔符版本 (tongueclicking)
        class_name.replace('_', ' '),  # 空格版本 (tongue clicking)
    ]
    return variants

def load_data_from_directory(data_dir, config):
    """从目录结构加载数据，支持多种目录命名格式"""
    audio_paths = []
    labels = []
    found_directories = {}
    
    print(f"Scanning data directory: {data_dir}")
    
    for class_name in config.class_names:
        class_label = config.class_to_idx[class_name]
        found_dir = None
        
        # 尝试不同的目录名变体
        variants = get_directory_variants(class_name)
        
        for variant in variants:
            test_dir = os.path.join(data_dir, variant)
            if os.path.exists(test_dir):
                found_dir = test_dir
                found_directories[class_name] = variant
                break
        
        if not found_dir:
            print(f"Warning: No directory found for class '{class_name}'. Tried: {variants}")
            continue
        
        print(f"Found directory for '{class_name}': {found_dir}")
        
        # 支持多种音频格式
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        class_file_count = 0
        for file_name in os.listdir(found_dir):
            # 跳过非音频文件和快捷方式
            if file_name.endswith('.lnk') or file_name.startswith('.'):
                continue
                
            if any(file_name.lower().endswith(ext) for ext in audio_extensions):
                audio_path = os.path.join(found_dir, file_name)
                
                # 检查文件是否可以正常访问
                try:
                    if os.path.getsize(audio_path) > 0:  # 确保文件不为空
                        audio_paths.append(audio_path)
                        labels.append(class_label)
                        class_file_count += 1
                except OSError:
                    print(f"Warning: Could not access file {audio_path}")
                    continue
        
        print(f"  Loaded {class_file_count} files for '{class_name}'")
    
    print(f"\nDirectory mapping:")
    for class_name, dir_name in found_directories.items():
        print(f"  {class_name} -> {dir_name}")
    
    return audio_paths, labels

def analyze_dataset_composition(audio_paths, labels, config):
    """分析数据集组成（真实数据 vs 合成数据）"""
    real_data_count = 0
    synthetic_data_count = 0
    
    for path in audio_paths:
        filename = os.path.basename(path)
        # 合成数据的文件名模式：class_name_number.wav
        if any(path.endswith(f"{class_name}_{i:03d}.wav") 
               for class_name in config.class_names 
               for i in range(1, 100)):  # 假设最多99个合成样本
            synthetic_data_count += 1
        else:
            real_data_count += 1
    
    print(f"\nDataset composition:")
    print(f"  Real data: {real_data_count} files")
    print(f"  Synthetic data: {synthetic_data_count} files")
    print(f"  Total: {len(audio_paths)} files")
    
    return real_data_count, synthetic_data_count

def create_data_loaders(config):
    """创建训练、验证和测试数据加载器"""
    
    # 加载数据
    audio_paths, labels = load_data_from_directory(config.data_dir, config)
    
    if len(audio_paths) == 0:
        print("No audio files found! Please add audio files to the data directory.")
        return None, None, None
    
    print(f"\nFound {len(audio_paths)} audio files")
    
    # 分析数据集组成
    analyze_dataset_composition(audio_paths, labels, config)
    
    # 按类别显示数据分布
    from collections import Counter
    label_counts = Counter(labels)
    print(f"\nClass distribution:")
    for idx, count in label_counts.items():
        class_name = config.idx_to_class[idx]
        print(f"  {class_name}: {count} files")
    
    # 检查类别平衡
    min_samples = min(label_counts.values())
    max_samples = max(label_counts.values())
    if max_samples / min_samples > 3:
        print(f"\nWarning: Dataset is imbalanced! "
              f"Max samples: {max_samples}, Min samples: {min_samples}")
        print("Consider adding more data for underrepresented classes.")
    
    # 确保每个类别都有足够的样本进行分割
    if min_samples < 3:
        print(f"\nError: Some classes have fewer than 3 samples. Cannot create train/val/test split.")
        print("Please add more data or adjust the split ratios.")
        return None, None, None
    
    # 分割数据集
    try:
        # 先分出训练集和临时集
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            audio_paths, labels, 
            test_size=(1 - config.train_split),
            random_state=42,
            stratify=labels
        )
        
        # 再从临时集分出验证集和测试集
        val_size = config.val_split / (config.val_split + config.test_split)
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(1 - val_size),
            random_state=42,
            stratify=temp_labels
        )
        
    except ValueError as e:
        print(f"\nError in data splitting: {e}")
        print("This usually happens when there's insufficient data for some classes.")
        print("Trying alternative splitting strategy...")
        
        # 备选方案：手动分割每个类别
        train_paths, val_paths, test_paths = [], [], []
        train_labels, val_labels, test_labels = [], [], []
        
        for class_idx in set(labels):
            class_paths = [p for p, l in zip(audio_paths, labels) if l == class_idx]
            class_labels = [l for l in labels if l == class_idx]
            
            n_total = len(class_paths)
            n_train = max(1, int(n_total * config.train_split))
            n_val = max(1, int(n_total * config.val_split))
            n_test = n_total - n_train - n_val
            
            if n_test < 1:
                n_test = 1
                n_val = max(1, n_total - n_train - n_test)
            
            train_paths.extend(class_paths[:n_train])
            val_paths.extend(class_paths[n_train:n_train + n_val])
            test_paths.extend(class_paths[n_train + n_val:])
            
            train_labels.extend([class_idx] * n_train)
            val_labels.extend([class_idx] * n_val)
            test_labels.extend([class_idx] * n_test)
    
    print(f"\nDataset split - Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # 创建数据集
    train_dataset = AudioDataset(train_paths, train_labels, config, is_training=True)
    val_dataset = AudioDataset(val_paths, val_labels, config, is_training=False)
    test_dataset = AudioDataset(test_paths, test_labels, config, is_training=False)
    
    # 自适应设置num_workers
    num_workers = min(4, len(train_paths) // config.batch_size) if len(train_paths) > config.batch_size else 0
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False  # 保留最后一个不完整的batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

def collate_fn(batch):
    """自定义批处理函数"""
    audios = torch.stack([item['audio'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    paths = [item['path'] for item in batch]
    
    return {
        'audio': audios,
        'label': labels,
        'paths': paths
    } 
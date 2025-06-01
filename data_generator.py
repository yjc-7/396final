import os
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import librosa

from config import Config, SOUND_DESCRIPTIONS

class AudioSampleGenerator:
    """音频样本生成器，用于生成示例训练数据"""
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.duration = config.max_audio_length
    
    def generate_laughing(self, variation: int = 0) -> np.ndarray:
        """生成笑声音频样本"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # 基础频率和谐波
        base_freq = 150 + variation * 20
        harmonics = [1, 2, 3, 4, 5]
        weights = [1, 0.5, 0.3, 0.2, 0.1]
        
        # 创建笑声的"哈哈哈"模式
        signal_wave = np.zeros_like(t)
        laugh_bursts = 3 + variation % 2  # 3-4个笑声爆发
        
        for burst in range(laugh_bursts):
            start_time = burst * (self.duration / laugh_bursts)
            end_time = start_time + 0.3  # 每个爆发持续0.3秒
            
            burst_mask = (t >= start_time) & (t <= end_time)
            burst_t = t[burst_mask] - start_time
            
            # 创建复合谐波
            burst_signal = np.zeros_like(burst_t)
            for harmonic, weight in zip(harmonics, weights):
                freq_modulation = 1 + 0.2 * np.sin(2 * np.pi * 5 * burst_t)  # 频率调制
                burst_signal += weight * np.sin(2 * np.pi * base_freq * harmonic * freq_modulation * burst_t)
            
            # 添加包络
            envelope = np.exp(-3 * burst_t) * (1 - np.exp(-20 * burst_t))
            signal_wave[burst_mask] = burst_signal * envelope
        
        # 添加噪音
        noise = np.random.normal(0, 0.05, len(signal_wave))
        signal_wave += noise
        
        # 归一化
        signal_wave = signal_wave / np.max(np.abs(signal_wave)) * 0.7
        
        return signal_wave
    
    def generate_sighing(self, variation: int = 0) -> np.ndarray:
        """生成叹气音频样本"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # 叹气的特征：先吸气，后呼气
        signal_wave = np.zeros_like(t)
        
        # 吸气阶段 (0-0.8秒)
        inhale_mask = t <= 0.8
        inhale_t = t[inhale_mask]
        
        # 创建吸气声（高频噪音）
        inhale_noise = np.random.normal(0, 1, len(inhale_t))
        inhale_filter = signal.butter(4, 2000, btype='high', fs=self.sample_rate, output='sos')
        inhale_filtered = signal.sosfilt(inhale_filter, inhale_noise)
        inhale_envelope = np.exp(-2 * inhale_t) * (1 - np.exp(-10 * inhale_t))
        signal_wave[inhale_mask] = inhale_filtered * inhale_envelope * 0.3
        
        # 短暂停顿 (0.8-1.0秒)
        
        # 呼气阶段 (1.0-3.5秒)
        exhale_mask = (t >= 1.0) & (t <= 3.5)
        exhale_t = t[exhale_mask] - 1.0
        
        # 创建呼气声（低频，类似"哎"声）
        base_freq = 120 + variation * 10
        exhale_signal = np.sin(2 * np.pi * base_freq * exhale_t)
        exhale_signal += 0.5 * np.sin(2 * np.pi * base_freq * 2 * exhale_t)
        
        # 添加沙哑质感
        noise = np.random.normal(0, 0.1, len(exhale_t))
        exhale_signal += noise
        
        # 呼气包络（缓慢衰减）
        exhale_envelope = np.exp(-0.5 * exhale_t)
        signal_wave[exhale_mask] = exhale_signal * exhale_envelope * 0.5
        
        # 归一化
        signal_wave = signal_wave / np.max(np.abs(signal_wave)) * 0.6
        
        return signal_wave
    
    def generate_tongue_clicking(self, variation: int = 0) -> np.ndarray:
        """生成咂舌音频样本"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_wave = np.zeros_like(t)
        
        # 咂舌声的特征：短促的爆破音
        num_clicks = 2 + variation % 3  # 2-4次咂舌
        
        for click in range(num_clicks):
            click_time = 0.5 + click * 0.8  # 间隔0.8秒
            
            # 每次咂舌持续约0.1秒
            click_start = click_time
            click_end = click_time + 0.1
            
            click_mask = (t >= click_start) & (t <= click_end)
            click_t = t[click_mask] - click_start
            
            # 创建短促的爆破声
            # 使用短脉冲和高频成分
            pulse = np.exp(-50 * click_t) * np.sin(2 * np.pi * 2000 * click_t)
            
            # 添加点击的"啪"声
            click_freq = 800 + variation * 100
            click_component = np.exp(-30 * click_t) * np.sin(2 * np.pi * click_freq * click_t)
            
            signal_wave[click_mask] = pulse + 0.5 * click_component
        
        # 添加轻微背景噪音
        noise = np.random.normal(0, 0.02, len(signal_wave))
        signal_wave += noise
        
        # 归一化
        signal_wave = signal_wave / np.max(np.abs(signal_wave)) * 0.8
        
        return signal_wave
    
    def generate_throat_clearing(self, variation: int = 0) -> np.ndarray:
        """生成清嗓音频样本"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_wave = np.zeros_like(t)
        
        # 清嗓的特征：类似"嗯咳"的声音
        
        # 第一部分：低沉的"嗯"声 (0-1秒)
        hum_mask = t <= 1.0
        hum_t = t[hum_mask]
        
        base_freq = 90 + variation * 15
        hum_signal = np.sin(2 * np.pi * base_freq * hum_t)
        hum_signal += 0.3 * np.sin(2 * np.pi * base_freq * 2 * hum_t)
        
        # 添加声带振动的质感
        modulation = 1 + 0.1 * np.sin(2 * np.pi * 8 * hum_t)
        hum_signal *= modulation
        
        hum_envelope = 1 - np.exp(-5 * hum_t)
        signal_wave[hum_mask] = hum_signal * hum_envelope * 0.4
        
        # 第二部分：清嗓的"咳"声 (1-2秒)
        cough_mask = (t >= 1.0) & (t <= 2.0)
        cough_t = t[cough_mask] - 1.0
        
        # 创建咳嗽的爆破音
        cough_noise = np.random.normal(0, 1, len(cough_t))
        cough_filter = signal.butter(4, [500, 3000], btype='band', fs=self.sample_rate, output='sos')
        cough_filtered = signal.sosfilt(cough_filter, cough_noise)
        
        cough_envelope = np.exp(-8 * cough_t) * (1 - np.exp(-20 * cough_t))
        signal_wave[cough_mask] = cough_filtered * cough_envelope * 0.6
        
        # 归一化
        signal_wave = signal_wave / np.max(np.abs(signal_wave)) * 0.7
        
        return signal_wave
    
    def generate_teeth_grinding(self, variation: int = 0) -> np.ndarray:
        """生成磨牙音频样本"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # 磨牙的特征：摩擦音，高频成分
        
        # 创建基础摩擦噪音
        noise = np.random.normal(0, 1, len(t))
        
        # 高通滤波，突出摩擦声
        highpass_filter = signal.butter(4, 1000, btype='high', fs=self.sample_rate, output='sos')
        filtered_noise = signal.sosfilt(highpass_filter, noise)
        
        # 添加周期性磨擦模式
        grinding_freq = 3 + variation * 0.5  # 磨牙频率
        grinding_pattern = 0.5 + 0.5 * np.sin(2 * np.pi * grinding_freq * t)
        
        # 添加强度变化
        intensity_pattern = 0.3 + 0.7 * np.sin(2 * np.pi * 0.5 * t)
        
        signal_wave = filtered_noise * grinding_pattern * intensity_pattern
        
        # 归一化
        signal_wave = signal_wave / np.max(np.abs(signal_wave)) * 0.5
        
        return signal_wave
    
    def generate_yawning(self, variation: int = 0) -> np.ndarray:
        """生成打哈欠音频样本"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_wave = np.zeros_like(t)
        
        # 打哈欠的特征：长的"啊"音，音调下降
        
        # 主要哈欠声 (0.5-3.5秒)
        yawn_mask = (t >= 0.5) & (t <= 3.5)
        yawn_t = t[yawn_mask] - 0.5
        
        # 基础频率，逐渐下降
        base_freq = 200 + variation * 20
        freq_decay = base_freq * np.exp(-0.3 * yawn_t)
        
        # 主音调
        yawn_signal = np.sin(2 * np.pi * np.cumsum(freq_decay) / self.sample_rate)
        
        # 添加谐波
        yawn_signal += 0.3 * np.sin(2 * np.pi * 2 * np.cumsum(freq_decay) / self.sample_rate)
        yawn_signal += 0.2 * np.sin(2 * np.pi * 3 * np.cumsum(freq_decay) / self.sample_rate)
        
        # 添加沙哑质感
        noise = np.random.normal(0, 0.1, len(yawn_t))
        yawn_signal += noise
        
        # 哈欠包络
        yawn_envelope = (1 - np.exp(-3 * yawn_t)) * np.exp(-0.5 * yawn_t)
        signal_wave[yawn_mask] = yawn_signal * yawn_envelope
        
        # 归一化
        signal_wave = signal_wave / np.max(np.abs(signal_wave)) * 0.6
        
        return signal_wave
    
    def generate_lip_smacking(self, variation: int = 0) -> np.ndarray:
        """生成抿嘴/咂嘴音频样本"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        signal_wave = np.zeros_like(t)
        
        # 抿嘴的特征：短促的"啪"声，类似亲吻声
        num_smacks = 2 + variation % 3  # 2-4次抿嘴
        
        for smack in range(num_smacks):
            smack_time = 1.0 + smack * 0.7  # 间隔0.7秒
            
            # 每次抿嘴持续约0.15秒
            smack_start = smack_time
            smack_end = smack_time + 0.15
            
            smack_mask = (t >= smack_start) & (t <= smack_end)
            smack_t = t[smack_mask] - smack_start
            
            # 创建抿嘴的"啪"声
            # 低频脉冲
            low_freq = 150 + variation * 20
            low_component = np.exp(-20 * smack_t) * np.sin(2 * np.pi * low_freq * smack_t)
            
            # 高频点击声
            high_freq = 1000 + variation * 200
            high_component = np.exp(-40 * smack_t) * np.sin(2 * np.pi * high_freq * smack_t)
            
            # 组合信号
            smack_signal = low_component + 0.3 * high_component
            
            signal_wave[smack_mask] = smack_signal
        
        # 添加轻微背景噪音
        noise = np.random.normal(0, 0.01, len(signal_wave))
        signal_wave += noise
        
        # 归一化
        signal_wave = signal_wave / np.max(np.abs(signal_wave)) * 0.7
        
        return signal_wave
    
    def generate_sample(self, class_name: str, variation: int = 0) -> np.ndarray:
        """根据类别名称生成音频样本"""
        generators = {
            'laughing': self.generate_laughing,
            'sighing': self.generate_sighing,
            'tongue_clicking': self.generate_tongue_clicking,
            'throat_clearing': self.generate_throat_clearing,
            'teeth_grinding': self.generate_teeth_grinding,
            'yawning': self.generate_yawning,
            'lip_smacking': self.generate_lip_smacking
        }
        
        if class_name not in generators:
            raise ValueError(f"Unknown class: {class_name}")
        
        return generators[class_name](variation)
    
    def generate_dataset(self, samples_per_class: int = 10):
        """生成完整的示例数据集"""
        print("Generating synthetic audio dataset...")
        
        total_generated = 0
        
        for class_name in self.config.class_names:
            class_dir = os.path.join(self.config.data_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            print(f"Generating {samples_per_class} samples for {class_name}...")
            
            for i in range(samples_per_class):
                # 生成音频
                audio_data = self.generate_sample(class_name, variation=i)
                
                # 保存文件
                filename = f"{class_name}_{i+1:03d}.wav"
                filepath = os.path.join(class_dir, filename)
                
                sf.write(filepath, audio_data, self.sample_rate)
                total_generated += 1
            
            print(f"✓ Generated {samples_per_class} samples for {class_name}")
        
        print(f"\nDataset generation completed!")
        print(f"Total samples generated: {total_generated}")
        print(f"Data saved to: {self.config.data_dir}")
        
        # 生成数据集说明
        self.generate_dataset_info()
    
    def generate_dataset_info(self):
        """生成数据集说明文件"""
        info = {
            "dataset_name": "Synthetic Sound Classification Dataset",
            "description": "人工生成的声音分类数据集，用于训练顾客行为声音识别模型",
            "classes": {},
            "audio_parameters": {
                "sample_rate": self.sample_rate,
                "duration": self.duration,
                "channels": 1,
                "format": "WAV"
            },
            "usage": {
                "training": "使用train.py脚本训练模型",
                "inference": "使用inference.py脚本进行预测",
                "data_format": "每个类别的音频文件存放在对应的子目录中"
            }
        }
        
        # 添加类别详细信息
        for class_name in self.config.class_names:
            info["classes"][class_name] = {
                "description": SOUND_DESCRIPTIONS.get(class_name, ""),
                "emotion": self.get_emotion_for_class(class_name),
                "business_meaning": self.get_business_meaning(class_name)
            }
        
        # 保存信息文件
        import json
        info_path = os.path.join(self.config.data_dir, "dataset_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset info saved to: {info_path}")
    
    def get_emotion_for_class(self, class_name: str) -> str:
        """获取类别对应的情绪"""
        from config import EMOTION_MAPPING
        return EMOTION_MAPPING.get(class_name, "neutral")
    
    def get_business_meaning(self, class_name: str) -> str:
        """获取类别的商业含义"""
        meanings = {
            "laughing": "顾客满意度高，体验良好的正面信号",
            "sighing": "顾客不满或不耐烦的负面信号，需要关注",
            "tongue_clicking": "顾客焦躁，可能对服务/产品不满",
            "throat_clearing": "顾客等待或紧张，可能需要帮助",
            "teeth_grinding": "强烈不满信号，需要立即处理",
            "yawning": "顾客缺乏兴趣或感到厌倦",
            "lip_smacking": "顾客期待或轻度满意的信号"
        }
        return meanings.get(class_name, "")

def main():
    """主函数"""
    config = Config()
    generator = AudioSampleGenerator(config)
    
    # 生成数据集
    samples_per_class = 20  # 每个类别生成20个样本
    generator.generate_dataset(samples_per_class)
    
    print("\n" + "="*50)
    print("数据集生成完成！")
    print("="*50)
    print("\n接下来的步骤：")
    print("1. 检查生成的数据：查看 data/ 目录")
    print("2. 如果有真实音频数据，请将其添加到相应的类别目录中")
    print("3. 运行训练：python train.py")
    print("4. 训练完成后，使用 inference.py 进行预测")

if __name__ == "__main__":
    main() 
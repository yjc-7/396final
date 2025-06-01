import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class Config:
    # 音频参数
    sample_rate: int = 16000
    max_audio_length: int = 5  # 秒
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    
    # 模型参数
    model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593"  # 预训练的AST模型
    num_classes: int = 7
    hidden_size: int = 768
    dropout: float = 0.1
    
    # 训练参数
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 50
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    
    # 数据参数
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # 路径设置
    data_dir: str = "data"
    model_save_dir: str = "models"
    log_dir: str = "logs"
    
    # 声音类别映射
    class_names: List[str] = None
    class_to_idx: Dict[str, int] = None
    idx_to_class: Dict[int, str] = None
    
    def __post_init__(self):
        # 定义声音类别
        self.class_names = [
            "laughing",      # 笑声 - 正向反馈
            "sighing",       # 叹气 - 负向情绪  
            "tongue_clicking", # 咂舌 - 焦躁/不耐烦
            "throat_clearing", # 清嗓 - 焦虑/紧张
            "teeth_grinding",  # 磨牙 - 挫败/生气
            "yawning",       # 打哈欠 - 厌烦/无聊
            "lip_smacking"   # 抿嘴/咂嘴 - 期待/轻度愉悦
        ]
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}
        
        # 创建必要的目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 为每个类别创建数据子目录
        for class_name in self.class_names:
            os.makedirs(os.path.join(self.data_dir, class_name), exist_ok=True)

# 情绪分类映射
EMOTION_MAPPING = {
    "laughing": "positive",
    "lip_smacking": "positive",
    "sighing": "negative", 
    "tongue_clicking": "negative",
    "throat_clearing": "neutral",
    "teeth_grinding": "negative",
    "yawning": "neutral"
}

# 声音描述
SOUND_DESCRIPTIONS = {
    "laughing": "顾客满意或享受体验时常会发出的正向反馈",
    "sighing": "等待过久、操作不顺或不满时表现出的负向情绪",
    "tongue_clicking": "对机器出错或结账流程不顺时，表达焦躁或不耐烦",
    "throat_clearing": "等待过程中尝试引起注意或心理紧张时会清嗓，通常伴随焦虑感", 
    "teeth_grinding": "强烈挫败或生气时下意识的紧张反应",
    "yawning": "大概率代表厌烦或无聊，可视为对系统体验不感兴趣",
    "lip_smacking": "对看见商品/结账成功感到期待、轻度愉悦时可能发出的声音"
} 
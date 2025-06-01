import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTForAudioClassification, ASTConfig, ASTFeatureExtractor
import librosa
import numpy as np

from config import Config

class SoundClassificationModel(nn.Module):
    def __init__(self, config):
        super(SoundClassificationModel, self).__init__()
        self.config = config
        
        # 使用预训练的AST模型
        self.feature_extractor = ASTFeatureExtractor.from_pretrained(config.model_name)
        
        # 加载预训练模型配置
        ast_config = ASTConfig.from_pretrained(config.model_name)
        ast_config.num_labels = config.num_classes
        
        # 初始化AST模型
        self.ast_model = ASTForAudioClassification.from_pretrained(
            config.model_name,
            config=ast_config,
            ignore_mismatched_sizes=True
        )
        
        # 添加dropout层提高泛化能力
        self.dropout = nn.Dropout(config.dropout)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(ast_config.hidden_size, ast_config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(ast_config.hidden_size // 2, config.num_classes)
        )
        
        # 冻结早期层（可选）
        self.freeze_early_layers()
    
    def freeze_early_layers(self, freeze_embeddings=True, freeze_early_encoders=True):
        """冻结模型的早期层以进行微调"""
        if freeze_embeddings:
            # 冻结位置嵌入和patch嵌入
            for param in self.ast_model.audio_spectrogram_transformer.embeddings.parameters():
                param.requires_grad = False
        
        if freeze_early_encoders:
            # 冻结前几层编码器
            num_layers_to_freeze = len(self.ast_model.audio_spectrogram_transformer.encoder.layer) // 2
            for i in range(num_layers_to_freeze):
                for param in self.ast_model.audio_spectrogram_transformer.encoder.layer[i].parameters():
                    param.requires_grad = False
    
    def forward(self, audio_input):
        """前向传播"""
        # 预处理音频为AST期望的格式
        processed_audio = self.preprocess_audio(audio_input)
        
        # 通过AST模型获取特征
        outputs = self.ast_model.audio_spectrogram_transformer(processed_audio)
        
        # 获取[CLS] token的表示
        sequence_output = outputs.last_hidden_state
        cls_output = sequence_output[:, 0]  # [CLS] token
        
        # 应用dropout
        cls_output = self.dropout(cls_output)
        
        # 分类
        logits = self.classifier(cls_output)
        
        return logits
    
    def preprocess_audio(self, audio_batch):
        """预处理音频批次为模型输入格式"""
        batch_size = audio_batch.shape[0]
        processed_batch = []
        
        for i in range(batch_size):
            audio = audio_batch[i].cpu().numpy()
            
            # 使用feature extractor处理音频
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            )
            
            processed_batch.append(inputs.input_values.squeeze())
        
        # 堆叠为批次
        processed_audio = torch.stack(processed_batch).to(audio_batch.device)
        
        return processed_audio

class SimpleCNN(nn.Module):
    """备用的简单CNN模型（如果AST不可用）"""
    def __init__(self, config):
        super(SimpleCNN, self).__init__()
        self.config = config
        
        # 卷积层
        self.conv_layers = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第四层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.5),
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, config.num_classes)
        )
    
    def forward(self, audio_input):
        # 将音频转换为Mel频谱图
        mel_specs = []
        for audio in audio_input:
            mel_spec = librosa.feature.melspectrogram(
                y=audio.cpu().numpy(),
                sr=self.config.sample_rate,
                n_mels=self.config.n_mels,
                n_fft=self.config.n_fft,
                hop_length=self.config.hop_length
            )
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            mel_specs.append(mel_spec)
        
        # 转换为tensor并添加通道维度
        mel_specs = torch.tensor(np.array(mel_specs), dtype=torch.float32).unsqueeze(1)
        mel_specs = mel_specs.to(audio_input.device)
        
        # 通过卷积层
        x = self.conv_layers(mel_specs)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 通过全连接层
        x = self.fc_layers(x)
        
        return x

def create_model(config, model_type="ast"):
    """创建模型的工厂函数"""
    if model_type == "ast":
        try:
            return SoundClassificationModel(config)
        except Exception as e:
            print(f"Failed to load AST model: {e}")
            print("Falling back to SimpleCNN model")
            return SimpleCNN(config)
    elif model_type == "cnn":
        return SimpleCNN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params 
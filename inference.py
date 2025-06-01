import os
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import argparse
import json

from config import Config, SOUND_DESCRIPTIONS, EMOTION_MAPPING
from model import create_model
from data_preprocessing import AudioDataset

class SoundClassifier:
    def __init__(self, model_path: str, config: Config = None):
        """
        初始化声音分类器
        
        Args:
            model_path: 模型文件路径
            config: 配置对象，如果为None则从模型文件加载
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载模型
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Error loading model: {e}")
            # 如果出现安全相关的错误，尝试使用safe_globals
            try:
                from config import Config
                torch.serialization.add_safe_globals([Config])
                checkpoint = torch.load(model_path, map_location=self.device)
            except:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if config is None:
            self.config = checkpoint['config']
        else:
            self.config = config
        
        # 创建模型
        self.model = create_model(self.config, model_type="ast")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {model_path}")
        print(f"Model supports {len(self.config.class_names)} classes: {', '.join(self.config.class_names)}")
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        预处理音频文件
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            预处理后的音频tensor
        """
        try:
            # 使用librosa加载音频
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            # 规范化音频长度
            target_length = self.config.sample_rate * self.config.max_audio_length
            
            if len(audio) > target_length:
                # 如果音频过长，截取中间部分
                start = (len(audio) - target_length) // 2
                audio = audio[start:start + target_length]
            elif len(audio) < target_length:
                # 如果音频过短，用零填充
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant', constant_values=0)
            
            # 归一化音频幅度
            audio = librosa.util.normalize(audio)
            
            # 转换为tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # 添加batch维度
            
            return audio_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # 返回静音
            silence = np.zeros(self.config.sample_rate * self.config.max_audio_length)
            return torch.tensor(silence, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def predict_single(self, audio_path: str) -> Dict:
        """
        对单个音频文件进行预测
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            预测结果字典
        """
        # 预处理音频
        audio_tensor = self.preprocess_audio(audio_path)
        
        # 预测
        with torch.no_grad():
            logits = self.model(audio_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class_idx = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        # 获取预测结果
        predicted_class = self.config.class_names[predicted_class_idx]
        emotion = EMOTION_MAPPING.get(predicted_class, "unknown")
        description = SOUND_DESCRIPTIONS.get(predicted_class, "无描述")
        
        # 所有类别的概率
        all_probabilities = {
            self.config.class_names[i]: probabilities[0][i].item()
            for i in range(len(self.config.class_names))
        }
        
        result = {
            'audio_path': audio_path,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'emotion': emotion,
            'description': description,
            'all_probabilities': all_probabilities
        }
        
        return result
    
    def predict_batch(self, audio_paths: List[str]) -> List[Dict]:
        """
        对多个音频文件进行批量预测
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            预测结果列表
        """
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.predict_single(audio_path)
                results.append(result)
                print(f"✓ {os.path.basename(audio_path)}: {result['predicted_class']} ({result['confidence']:.3f})")
            except Exception as e:
                print(f"✗ Error processing {audio_path}: {e}")
                results.append({
                    'audio_path': audio_path,
                    'error': str(e)
                })
        
        return results
    
    def predict_directory(self, directory_path: str) -> List[Dict]:
        """
        对目录中的所有音频文件进行预测
        
        Args:
            directory_path: 目录路径
            
        Returns:
            预测结果列表
        """
        # 支持的音频格式
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        
        # 收集所有音频文件
        audio_paths = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_paths.append(os.path.join(root, file))
        
        print(f"Found {len(audio_paths)} audio files in {directory_path}")
        
        if not audio_paths:
            print("No audio files found!")
            return []
        
        return self.predict_batch(audio_paths)
    
    def visualize_predictions(self, results: List[Dict], save_path: str = None):
        """
        可视化预测结果
        
        Args:
            results: 预测结果列表
            save_path: 保存路径，如果为None则显示图片
        """
        # 统计预测类别
        predictions = [r['predicted_class'] for r in results if 'predicted_class' in r]
        
        if not predictions:
            print("No valid predictions to visualize")
            return
        
        from collections import Counter
        prediction_counts = Counter(predictions)
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测类别分布
        classes = list(prediction_counts.keys())
        counts = list(prediction_counts.values())
        
        axes[0, 0].bar(classes, counts, color='skyblue')
        axes[0, 0].set_title('Predicted Class Distribution')
        axes[0, 0].set_xlabel('Sound Class')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. 置信度分布
        confidences = [r['confidence'] for r in results if 'confidence' in r]
        axes[0, 1].hist(confidences, bins=20, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Confidence Distribution')
        axes[0, 1].set_xlabel('Confidence')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(np.mean(confidences), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(confidences):.3f}')
        axes[0, 1].legend()
        
        # 3. 情绪分布
        emotions = [r['emotion'] for r in results if 'emotion' in r]
        emotion_counts = Counter(emotions)
        
        axes[1, 0].pie(emotion_counts.values(), labels=emotion_counts.keys(), 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Emotion Distribution')
        
        # 4. 置信度箱线图（按类别）
        class_confidences = {}
        for result in results:
            if 'predicted_class' in result and 'confidence' in result:
                class_name = result['predicted_class']
                if class_name not in class_confidences:
                    class_confidences[class_name] = []
                class_confidences[class_name].append(result['confidence'])
        
        if class_confidences:
            box_data = [class_confidences[cls] for cls in class_confidences.keys()]
            axes[1, 1].boxplot(box_data, labels=list(class_confidences.keys()))
            axes[1, 1].set_title('Confidence by Class')
            axes[1, 1].set_xlabel('Sound Class')
            axes[1, 1].set_ylabel('Confidence')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        保存预测结果到JSON文件
        
        Args:
            results: 预测结果列表
            output_path: 输出文件路径
        """
        # 计算统计信息
        valid_results = [r for r in results if 'predicted_class' in r]
        
        summary = {
            'total_files': len(results),
            'successful_predictions': len(valid_results),
            'failed_predictions': len(results) - len(valid_results),
            'class_distribution': {},
            'emotion_distribution': {},
            'average_confidence': 0.0
        }
        
        if valid_results:
            from collections import Counter
            
            # 类别分布
            classes = [r['predicted_class'] for r in valid_results]
            summary['class_distribution'] = dict(Counter(classes))
            
            # 情绪分布
            emotions = [r['emotion'] for r in valid_results]
            summary['emotion_distribution'] = dict(Counter(emotions))
            
            # 平均置信度
            confidences = [r['confidence'] for r in valid_results]
            summary['average_confidence'] = float(np.mean(confidences))
        
        # 准备输出数据
        output_data = {
            'summary': summary,
            'results': results,
            'class_descriptions': SOUND_DESCRIPTIONS
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {output_path}")
        print(f"Summary: {summary['successful_predictions']}/{summary['total_files']} files processed successfully")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='声音分类推理')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径')
    parser.add_argument('--input', type=str, required=True,
                       help='输入音频文件或目录路径')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='输出结果文件路径')
    parser.add_argument('--visualize', action='store_true',
                       help='是否生成可视化图表')
    parser.add_argument('--viz_output', type=str, default='predictions_viz.png',
                       help='可视化图表保存路径')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    # 创建分类器
    classifier = SoundClassifier(args.model_path)
    
    # 执行预测
    print(f"\nProcessing: {args.input}")
    
    if os.path.isfile(args.input):
        # 单个文件
        results = [classifier.predict_single(args.input)]
        print(f"\nPrediction for {os.path.basename(args.input)}:")
        result = results[0]
        print(f"Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Emotion: {result['emotion']}")
        print(f"Description: {result['description']}")
        
        # 显示所有类别的概率
        print("\nAll class probabilities:")
        for class_name, prob in sorted(result['all_probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {prob:.3f}")
    
    elif os.path.isdir(args.input):
        # 目录
        results = classifier.predict_directory(args.input)
    else:
        print(f"Input path not found: {args.input}")
        return
    
    # 保存结果
    classifier.save_results(results, args.output)
    
    # 生成可视化
    if args.visualize and len(results) > 1:
        classifier.visualize_predictions(results, args.viz_output)

if __name__ == "__main__":
    main() 
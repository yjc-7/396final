#!/usr/bin/env python3
"""
真实数据训练脚本
专门针对真实音频数据优化的训练流程
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
import argparse

from config import Config, SOUND_DESCRIPTIONS
from data_preprocessing import create_data_loaders
from model import create_model, count_parameters

class RealDataTrainer:
    def __init__(self, config, use_wandb=False):
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 初始化wandb（如果需要）
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project="sound-classification-real-data",
                    config=vars(config),
                    name=f"real_data_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
            except ImportError:
                print("Warning: wandb not installed. Continuing without wandb logging.")
                self.use_wandb = False
        
        # 创建数据加载器
        print("Creating data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(config)
        
        if self.train_loader is None:
            raise ValueError("No data found! Please run organize_data.py first.")
        
        # 创建模型
        print("Creating model...")
        try:
            self.model = create_model(config, model_type="ast")
        except Exception as e:
            print(f"Failed to load AST model: {e}")
            print("Falling back to CNN model...")
            self.model = create_model(config, model_type="cnn")
        
        self.model.to(self.device)
        
        # 显示模型参数
        count_parameters(self.model)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        
        # 为真实数据优化的学习率
        if hasattr(self.model, 'ast_model'):  # AST模型
            # 对预训练模型使用较低的学习率
            self.optimizer = optim.AdamW([
                {'params': self.model.ast_model.parameters(), 'lr': config.learning_rate * 0.1},
                {'params': self.model.classifier.parameters(), 'lr': config.learning_rate}
            ], weight_decay=config.weight_decay)
        else:  # CNN模型
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # 学习率调度器 - 对真实数据使用更保守的调度
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        
        # 早停
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.patience = 15  # 对真实数据增加耐心
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            try:
                logits = self.model(audio)
                loss = self.criterion(logits, labels)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_val
                )
                
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"Error in batch processing: {e}")
                continue
            
            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.6f}"
            })
        
        # 计算指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                try:
                    logits = self.model(audio)
                    loss = self.criterion(logits, labels)
                    
                    total_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self):
        """主训练循环"""
        print("Starting training with real data...")
        print(f"Training on {len(self.train_loader.dataset)} samples")
        print(f"Validating on {len(self.val_loader.dataset)} samples")
        print(f"Testing on {len(self.test_loader.dataset)} samples")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            # 更新学习率（基于验证损失）
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_acc)
            self.train_history['learning_rate'].append(current_lr)
            
            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {current_lr:.6f}")
            
            # wandb记录
            if self.use_wandb:
                import wandb
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr
                })
            
            # 保存最佳模型（基于准确率）
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model('best_real_data_model.pth')
                print(f"New best model saved! Val Acc: {val_acc:.4f}")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
            
            # 每5个epoch保存一次检查点
            if (epoch + 1) % 5 == 0:
                self.save_model(f'checkpoint_real_data_epoch_{epoch + 1}.pth')
        
        # 训练结束后保存最终模型
        self.save_model('final_real_data_model.pth')
        
        # 在测试集上评估
        print("\nEvaluating on test set...")
        test_metrics = self.evaluate_test_set()
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        return self.train_history
    
    def save_model(self, filename):
        """保存模型"""
        model_path = os.path.join(self.config.model_save_dir, filename)
        
        # 保存模型状态
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'class_names': self.config.class_names
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    def evaluate_test_set(self):
        """在测试集上评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                try:
                    logits = self.model(audio)
                    predictions = torch.argmax(logits, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"Error in test batch: {e}")
                    continue
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # 分类报告
        class_names = self.config.class_names
        report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 保存结果
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'true_labels': all_labels,
            'model_info': {
                'total_real_data_samples': len(self.train_loader.dataset) + len(self.val_loader.dataset) + len(self.test_loader.dataset),
                'train_samples': len(self.train_loader.dataset),
                'val_samples': len(self.val_loader.dataset),
                'test_samples': len(self.test_loader.dataset)
            }
        }
        
        # 保存到文件
        results_path = os.path.join(self.config.log_dir, 'real_data_test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm, class_names, save_name='real_data_confusion_matrix.png')
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names, save_name='confusion_matrix.png'):
        """绘制混淆矩阵"""
        plt.figure(figsize=(12, 10))
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建标注
        annot = np.array([[f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)' 
                          for j in range(cm.shape[1])] 
                         for i in range(cm.shape[0])])
        
        sns.heatmap(
            cm_percent, 
            annot=annot,
            fmt='',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            square=True
        )
        
        plt.title('Confusion Matrix - Real Data Results', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存图片
        save_path = os.path.join(self.config.log_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_history['loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(self.train_history['val_loss'], label='Val Loss', color='red')
        axes[0, 0].set_title('Loss - Real Data Training', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.train_history['accuracy'], label='Train Accuracy', color='blue')
        axes[0, 1].plot(self.train_history['val_accuracy'], label='Val Accuracy', color='red')
        axes[0, 1].set_title('Accuracy - Real Data Training', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 0].plot(self.train_history['learning_rate'], color='green')
        axes[1, 0].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 性能总结
        axes[1, 1].axis('off')
        if self.train_history['val_accuracy']:
            best_val_acc = max(self.train_history['val_accuracy'])
            best_epoch = self.train_history['val_accuracy'].index(best_val_acc) + 1
            final_val_acc = self.train_history['val_accuracy'][-1]
            
            summary_text = f"""
            Real Data Training Summary:
            
            Best Validation Accuracy: {best_val_acc:.4f}
            Best Epoch: {best_epoch}
            Final Validation Accuracy: {final_val_acc:.4f}
            Total Epochs: {len(self.train_history['loss'])}
            
            Dataset Info:
            Total Samples: {len(self.train_loader.dataset) + len(self.val_loader.dataset) + len(self.test_loader.dataset)}
            Train: {len(self.train_loader.dataset)}
            Val: {len(self.val_loader.dataset)}
            Test: {len(self.test_loader.dataset)}
            """
            
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, 
                           verticalalignment='center', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('Real Data Training Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(self.config.log_dir, 'real_data_training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training history saved to {save_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="真实数据训练")
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--use_wandb', action='store_true', help='使用wandb记录')
    parser.add_argument('--model_type', choices=['ast', 'cnn'], default='ast', help='模型类型')
    
    args = parser.parse_args()
    
    # 创建配置
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    
    print("="*60)
    print("🎵 REAL DATA SOUND CLASSIFICATION TRAINING")
    print("="*60)
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Model Type: {args.model_type}")
    print("="*60)
    
    # 创建训练器
    trainer = RealDataTrainer(config, use_wandb=args.use_wandb)
    
    # 开始训练
    history = trainer.train()
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    print("="*60)
    print("🎉 REAL DATA TRAINING COMPLETED!")
    print("="*60)
    print("Next steps:")
    print("1. Check the training results in logs/")
    print("2. Use the trained model for inference:")
    print("   py inference.py --model_path models/best_real_data_model.pth --input your_audio.wav")
    print("3. Analyze the confusion matrix and classification report")

if __name__ == "__main__":
    main() 
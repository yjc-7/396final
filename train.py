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
import wandb
import json
from datetime import datetime

from config import Config, SOUND_DESCRIPTIONS
from data_preprocessing import create_data_loaders
from model import create_model, count_parameters

class Trainer:
    def __init__(self, config, use_wandb=False):
        self.config = config
        self.use_wandb = use_wandb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 初始化wandb
        if self.use_wandb:
            wandb.init(
                project="sound-classification",
                config=vars(config),
                name=f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # 创建数据加载器
        print("Creating data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(config)
        
        if self.train_loader is None:
            raise ValueError("No data found! Please add audio files to the data directory.")
        
        # 创建模型
        print("Creating model...")
        self.model = create_model(config, model_type="ast")
        self.model.to(self.device)
        
        # 显示模型参数
        count_parameters(self.model)
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=1e-6
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
        self.patience_counter = 0
        self.patience = 10
    
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
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
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
                
                logits = self.model(audio)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_labels
    
    def train(self):
        """主训练循环"""
        print("Starting training...")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            # 更新学习率
            self.scheduler.step()
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
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr
                })
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model('best_model.pth')
                print("Best model saved!")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                self.save_model(f'checkpoint_epoch_{epoch + 1}.pth')
        
        # 训练结束后保存最终模型
        self.save_model('final_model.pth')
        
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
            'best_val_loss': self.best_val_loss
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename):
        """加载模型"""
        model_path = os.path.join(self.config.model_save_dir, filename)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Model loaded from {model_path}")
    
    def evaluate_test_set(self):
        """在测试集上评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(audio)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # 分类报告
        class_names = self.config.class_names
        report = classification_report(
            all_labels, 
            all_predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        # 保存结果
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': all_predictions,
            'true_labels': all_labels
        }
        
        # 保存到文件
        results_path = os.path.join(self.config.log_dir, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm, class_names)
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(os.path.join(self.config.log_dir, 'confusion_matrix.png'), dpi=300)
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.train_history['loss'], label='Train Loss')
        axes[0, 0].plot(self.train_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.train_history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(self.train_history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[1, 0].plot(self.train_history['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Class Distribution (if available)
        if hasattr(self, 'class_distribution'):
            axes[1, 1].bar(self.config.class_names, self.class_distribution)
            axes[1, 1].set_title('Class Distribution')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, 'training_history.png'), dpi=300)
        plt.show()

def main():
    """主函数"""
    # 创建配置
    config = Config()
    
    # 创建训练器
    trainer = Trainer(config, use_wandb=False)  # 设置为True以使用wandb
    
    # 开始训练
    history = trainer.train()
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    print("Training completed!")

if __name__ == "__main__":
    main() 
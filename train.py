#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2025/03/26 10:17:50
@Author  :   Neutrin 
'''

# here put the import lib
import numpy as np
import pandas as pd
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os 
import cv2 
from torchvision import transforms
import random
import warnings
from tqdm import tqdm
import argparse
from datetime import datetime

# 导入自定义模块
from data.data_interface import BreastCancerDataset, DInterface
from model.model_interface import MInterface

# 忽略警告
warnings.filterwarnings('ignore')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='乳腺癌分类训练脚本')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='E:/Dataset/breastcancer2',
                        help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='训练批量大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'resnet50', 'efficientnet_b0', 'mobilenetv2_100'],
                        help='模型类型')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类类别数')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout比率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减参数')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, save_dir='checkpoints'):
    """
    训练模型函数
    
    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        num_epochs: 训练轮数
        save_dir: 模型保存目录
    
    返回:
        model: 训练后的模型
        history: 训练历史记录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录训练开始时间
    start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    best_acc = 0.0
    best_model_path = os.path.join(save_dir, f'best_model_{start_time}.pth')
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # 使用tqdm显示训练进度
    for epoch in tqdm(range(num_epochs), desc="训练进度"):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # 使用tqdm显示批次进度
            with tqdm(dataloader, desc=f"{phase}", leave=False) as batch_pbar:
                for inputs, labels in batch_pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # 清零梯度
                    optimizer.zero_grad()
                    
                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # 反向传播与优化（仅在训练阶段）
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # 统计损失和准确率
                    batch_loss = loss.item() * inputs.size(0)
                    batch_corrects = torch.sum(preds == labels.data)
                    running_loss += batch_loss
                    running_corrects += batch_corrects
                    
                    # 更新批次进度条
                    batch_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{batch_corrects.double() / inputs.size(0):.4f}'
                    })
            
            # 计算当前epoch的平均损失和准确率
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 保存训练记录
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # 使用学习率调度器
                scheduler.step(epoch_loss)
                
                # 只保存最佳模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)
                    print(f"发现更好的模型，准确率: {epoch_acc:.4f}")
    
    # 训练结束后，加载最佳模型权重
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"训练完成，加载最佳模型权重 (准确率: {best_acc:.4f})")
    
    return model, history

def visualize_training_history(history):
    """
    可视化训练历史
    
    参数:
        history: 包含训练历史的字典
    """
    plt.figure(figsize=(12, 4))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练与验证损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.title('训练与验证准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据接口
    data_interface = DInterface(
        root_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 获取数据加载器
    train_loader = data_interface.train_dataloader()
    val_loader = data_interface.val_dataloader()
    test_loader = data_interface.test_dataloader()
    
    # 显示数据集信息
    dataset_info = data_interface.get_dataset_info()
    print("\n数据集信息:")
    print(f"- 训练集大小: {dataset_info['dataset_sizes']['train']}")
    print(f"- 验证集大小: {dataset_info['dataset_sizes']['val']}")
    print(f"- 测试集大小: {dataset_info['dataset_sizes']['test']}")
    print(f"- 类别分布：")
    for class_name, count in dataset_info['class_distribution']['train'].items():
        print(f"  * {class_name}: {count}")
    
    # 创建模型
    model_interface = MInterface(
        num_classes=args.num_classes,
        device=device
    )
    model = model_interface[0]
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5
    )
    
    # 训练模型
    print(f"\n开始训练模型...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=10,
        save_dir=args.save_dir
    )
    
    # 可视化训练历史
    visualize_training_history(history)
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到 {final_model_path}")
    
    # 在测试集上评估模型
    print("\n在测试集上评估模型...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试评估"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            test_correct += torch.sum(preds == labels.data)
    
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct.double() / len(test_loader.dataset)
    
    print(f"测试集结果 - 损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")

if __name__ == "__main__":
    main()





#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fine_tuning.py
@Time    :   2025/04/24 19:19:38
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
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
# 导入自定义模块
from data.data_interface import ImageDataset, DInterface
from model.model_interface import MInterface, get_available_wavelets
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 修改参数解析函数以添加迁移学习相关参数
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='乳腺癌分类训练脚本')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='D:\\Dataset\\mini-imagenet\\Mini-ImageNet-Dataset',
                        help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='训练批量大小')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='数据加载线程数')
    
    # 模型参数
    parser.add_argument('--model_type', type=str, default='cnn',
                        choices=['cnn', 'waveletcnn', 'efficientnet-b0', 'mobilenetv2_100','unet', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                        'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7','efficientnet-b8','vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn','resnet18',
                        'resnet34','resnet50','resnet101','resnet152','GoogleNet', 'efficientnet-cbam-b0', 'efficientnet-cbam-b1', 'efficientnet-cbam-b2', 'efficientnet-cbam-b3',
                        'efficientnet-cbam-b4', 'efficientnet-cbam-b5', 'efficientnet-cbam-b6', 'efficientnet-cbam-b7','efficientnet-cbam-b8','effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl',
                        'effnetv2_cbam_s', 'effnetv2_cbam_m', 'effnetv2_cbam_l', 'effnetv2_cbam_xl','effnetv2_dwt_cbam_s', 'effnetv2_dwt_cbam_m', 'effnetv2_dwt_cbam_l', 'effnetv2_dwt_cbam_xl'],
                        help='模型类型')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='分类类别数')
    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='Dropout比率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='权重衰减参数')

    # 迁移学习参数
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='预训练模型权重路径')
    parser.add_argument('--freeze_layers', action='store_true',
                       help='是否冻结模型早期层')
    parser.add_argument('--finetune_lr', type=float, default=1e-4,
                       help='微调学习率（通常比从头训练小）')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')

    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='模型保存目录')
    
    # 学习率调度器参数
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                      choices=['cosine', 'reduce_plateau', 'step', 'multi_step'],
                      help='学习率调度器类型')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                      help='余弦退火的最小学习率')

    return parser.parse_args()

# 设置随机种子
def set_seed(seed):
    """设置随机种子以确保结果可重复"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 添加一个加载预训练模型的函数
def load_pretrained_model(model_interface, model_type, pretrained_path, num_classes):
    """
    加载预训练模型并修改分类层以适应新任务
    
    参数:
        model_interface (MInterface): 模型接口对象
        model_type (str): 模型类型
        pretrained_path (str): 预训练模型权重路径
        num_classes (int): 目标任务的类别数
        
    返回:
        nn.Module: 准备好进行迁移学习的模型
    """
    # 初始化模型，带有预训练权重
    model = model_interface.create_model(model_type=model_type, pretrained=False)
    
    # 加载预训练权重
    if os.path.exists(pretrained_path):
        # 加载预训练权重
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # 有时预训练权重的state_dict可能在model_state_dict或state_dict键下
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        # 有些预训练模型的权重键可能有前缀，需要处理
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        # 查找并替换分类层权重
        model_dict = model.state_dict()
        # 只加载匹配的层
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # 更新模型权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"已加载预训练权重，共计 {len(pretrained_dict)}/{len(model_dict)} 层")
    else:
        print(f"未找到预训练模型：{pretrained_path}")
        
    # 修改分类层以适应新任务
    model_interface.modify_classifier(model, num_classes)
    
    return model

# 添加冻结层函数
def freeze_layers(model, freeze_until=None, freeze_names=None):
    """
    冻结模型的指定层
    
    参数:
        model (nn.Module): 要冻结层的模型
        freeze_until (str): 冻结直到指定名称的层（比如'layer3'）
        freeze_names (list): 冻结包含指定名称的层列表
    """
    freeze_state = False   
    
    # 计算可训练参数总数
    total_params = sum(p.numel() for p in model.parameters())  
    
    for name, param in model.named_parameters():
        # 如果指定了freeze_until，则冻结该层之前的所有层
        if freeze_until is not None:
            if freeze_until in name:
                freeze_state = True
            param.requires_grad = freeze_state
            
        # 如果指定了freeze_names，则冻结包含这些名称的层
        elif freeze_names is not None:
            param.requires_grad = not any(layer_name in name for layer_name in freeze_names)
    
    # 计算可训练的参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型总参数：{total_params:,}")
    print(f"可训练参数：{trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"冻结参数：{total_params-trainable_params:,} ({(total_params-trainable_params)/total_params*100:.2f}%)")
    
    return model

# 迁移学习函数
def transfer_learning(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, save_dir='checkpoints'):
    """
    迁移学习训练函数
    
    参数:
        model (nn.Module): 准备好的预训练模型
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
    # 设置模型为训练模式
    scaler = torch.amp.GradScaler()    

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
    
    # 添加学习率历史记录
    history['learning_rate'] = []
    
    # 使用tqdm显示训练进度
    for epoch in tqdm(range(num_epochs), desc="训练进度"):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        print(f'当前学习率: {current_lr:.6f}')

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
                    labels = torch.tensor(labels, dtype=torch.long, device=device)
                    
                    # 清零梯度
                    optimizer.zero_grad()
                    
                    if phase == 'train':
                        # 训练阶段使用混合精度
                        with autocast(device_type='cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        # 使用scaler进行反向传播和优化
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        with torch.no_grad():
                            with autocast(device_type='cuda'):
                                # 验证阶段使用混合精度
                                outputs = model(inputs)
                                loss = criterion(outputs, labels)
                    
                    # 获取预测结果
                    _, preds = torch.max(outputs, 1)

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
                
                # 仅在验证阶段结束后调用一次学习率调度器
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(epoch_loss)  # ReduceLROnPlateau需要监控值
                else:
                    scheduler.step()  # 其他调度器无需参数

                # 只保存最佳模型
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_acc': best_acc,
                        'history': history
                    }, best_model_path)
                    print(f"发现更好的模型，准确率: {epoch_acc:.4f}")

    # 训练结束后，加载最佳模型权重
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"训练完成，加载最佳模型权重 (准确率: {best_acc:.4f})")
    
    return model, history

# 添加主函数，整合完整的迁移学习流程
def fine_tune_model_from_pretrained(args):
    """
    从预训练模型开始进行迁移学习的主流程
    
    参数:
        args: 命令行参数
    
    返回:
        model: 微调后的模型
        history: 训练历史
    """
    # 设置随机种子
    set_seed(args.seed)
    
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化数据接口
    data_interface = DInterface(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # 获取数据加载器
    train_loader = data_interface.train_dataloader()
    val_loader = data_interface.val_dataloader()
    
    # 初始化模型接口
    model_interface = MInterface()
    
    # 加载预训练模型并准备迁移学习
    model = load_pretrained_model(
        model_interface=model_interface,
        model_type=args.model_type,
        pretrained_path=args.pretrained_path,  # 需要在parse_args中添加此参数
        num_classes=args.num_classes
    )
    
    # 冻结模型早期层
    if args.freeze_layers:  # 需要在parse_args中添加此参数
        # 根据模型类型选择冻结策略
        if 'efficientnet' in args.model_type:
            # 冻结特征提取器，只训练分类器
            model = freeze_layers(model, freeze_names=['features'])
        elif 'resnet' in args.model_type:
            # 冻结前面几个层
            model = freeze_layers(model, freeze_until='layer3')
        else:
            # 默认冻结70%的层
            model = freeze_layers(model, freeze_until=None, freeze_names=None)
    
    # 将模型移至设备
    model = model.to(device)
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 创建优化器 - 对于微调，通常使用较小的学习率
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 创建学习率调度器
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.lr_scheduler == 'reduce_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=args.min_lr
        )
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    elif args.lr_scheduler == 'multi_step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[30, 60, 90], gamma=0.1
        )
    
    # 进行迁移学习训练
    model, history = transfer_learning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    return model, history

# 添加可视化训练历史函数
def visualize_training_history(history, model_type):
    """
    可视化训练历史
    
    参数:
        history: 包含训练历史的字典
        model_type: 模型类型名称
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
    # 使用时间戳创建唯一的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'{model_type}_训练曲线_{timestamp}.png'
    plt.savefig(save_path)
    print(f"训练历史图表已保存至: {save_path}")
    plt.show()

    if 'learning_rate' in history:
        plt.figure(figsize=(10, 4))
        plt.plot(history['learning_rate'], 'o-')
        plt.xlabel('轮次')
        plt.ylabel('学习率')
        plt.title('学习率变化曲线')
        plt.yscale('log')  # 使用对数刻度更清晰地显示变化
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        lr_save_path = f'{model_type}_学习率曲线_{timestamp}.png'
        plt.savefig(lr_save_path)
        print(f"学习率变化曲线已保存至: {lr_save_path}")
        plt.show()


# 添加保存训练历史到CSV的函数
def save_training_history_to_csv(history, model_type, filename=None, save_dir='results'):
    """
    将训练历史保存为CSV文件
    
    参数:
        history: 包含训练历史的字典
        model_type: 模型类型
        filename: 保存的文件名（如果为None，则使用时间戳生成）
        save_dir: 保存文件的目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{model_type}_finetune_训练历史数据_{timestamp}.csv'
    
    # 创建完整的文件路径
    filepath = os.path.join(save_dir, filename)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'train_acc': history['train_acc'],
        'val_loss': history['val_loss'],
        'val_acc': history['val_acc']
    })
    
    # 如果有学习率历史，也添加到DataFrame中
    if 'learning_rate' in history:
        df['learning_rate'] = history['learning_rate']
    
    # 保存为CSV文件
    df.to_csv(filepath, index=False)
    print(f"训练历史数据已保存到: {filepath}")


# 添加ROC曲线绘制和AUC值保存函数
def plot_roc_curve(y_true, y_pred_proba, model_type, class_names=None, save_dir='results'):
    """
    绘制ROC曲线并计算AUC
    
    参数:
        y_true: 真实标签
        y_pred_proba: 预测概率
        model_type: 模型类型名称
        class_names: 类别名称列表
        save_dir: 保存图像的目录
    
    返回:
        auc_values: 每个类别的AUC值字典
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 转换为numpy数组
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    n_classes = y_pred_proba.shape[1]
    
    # 如果没有提供类别名称，则使用数字作为类别名称
    if class_names is None:
        class_names = [f'类别 {i}' for i in range(n_classes)]
    
    # 二值化标签（one-hot编码）
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # 计算ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(10, 8))
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown', 'pink', 'gray', 'olive'])
    
    # 对每个类别计算ROC曲线和AUC
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        if i < 10:  # 只显示前10个类别，避免图像过于拥挤
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
    
    # 计算宏平均AUC（所有类别的平均）
    if n_classes > 2:
        try:
            macro_auc = roc_auc_score(y_true_bin, y_pred_proba, average='macro')
            print(f"宏平均AUC: {macro_auc:.4f}")
        except:
            print("计算宏平均AUC出错，可能是某些类别缺少正样本或负样本")
            macro_auc = np.mean(list(roc_auc.values()))
    else:
        macro_auc = list(roc_auc.values())[0]
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (False Positive Rate)')
    plt.ylabel('真阳性率 (True Positive Rate)')
    plt.title(f'ROC曲线 - {model_type} 微调模型 (宏平均AUC = {macro_auc:.4f})')
    plt.legend(loc="lower right")
    
    # 保存图像
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(save_dir, f'{model_type}_finetune_ROC曲线_{timestamp}.png')
    plt.savefig(save_path)
    print(f"ROC曲线已保存至: {save_path}")
    plt.show()
    
    # 返回每个类别的AUC值
    auc_values = {class_names[i]: roc_auc[i] for i in roc_auc}
    auc_values['宏平均'] = macro_auc
    
    return auc_values


# 保存测试评估结果的函数
def save_evaluation_results(metrics, model_type, save_dir='results'):
    """
    保存模型评估指标到CSV文件
    
    参数:
        metrics: 包含各项指标的字典
        model_type: 模型类型名称
        save_dir: 保存目录
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建DataFrame
    metrics_df = pd.DataFrame({
        'metric': list(metrics.keys()),
        'value': list(metrics.values())
    })
    
    # 保存文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(save_dir, f'{model_type}_finetune_评估指标_{timestamp}.csv')
    metrics_df.to_csv(file_path, index=False)
    
    print(f"评估指标已保存到: {file_path}")
    
    # 同时保存为TXT报告
    report_path = os.path.join(save_dir, f'{model_type}_finetune_评估报告_{timestamp}.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"模型: {model_type} (微调)\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("评估指标:\n")
        
        for metric, value in metrics.items():
            f.write(f"- {metric}: {value:.4f}\n")
    
    print(f"评估报告已保存到: {report_path}")



# 评估微调后模型的函数
def evaluate_finetuned_model(model, test_loader, criterion, device, model_type):
    """评估微调后的模型并保存结果"""
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_top5_correct = 0
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试评估"):
            inputs = inputs.to(device)
            labels = torch.tensor(labels, dtype=torch.long, device=device)
            
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)
            
            # 计算Top-1预测
            _, preds = torch.max(outputs, 1)
            
            # 计算Top-5预测（当类别数大于5时）
            if outputs.size(1) >= 5:
                _, top5_preds = torch.topk(outputs, k=min(5, outputs.size(1)), dim=1)
                batch_top5_correct = sum([l.item() in p.tolist() for l, p in zip(labels, top5_preds)])
                test_top5_correct += batch_top5_correct
            
            test_loss += loss.item() * inputs.size(0)
            test_correct += torch.sum(preds == labels.data).item()
            
            # 收集标签和预测
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # 计算各项指标
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct / len(test_loader.dataset)
    
    if outputs.size(1) >= 5:
        test_top5_acc = test_top5_correct / len(test_loader.dataset)
    else:
        test_top5_acc = test_acc  # 类别数少于5时，使用top-1准确率
    
    # 计算其他性能指标
    test_precision_macro = precision_score(all_labels, all_preds, average='macro')
    test_precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    test_recall_macro = recall_score(all_labels, all_preds, average='macro')
    test_recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    test_f1_macro = f1_score(all_labels, all_preds, average='macro')
    test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # 整理所有指标
    metrics = {
        'accuracy': test_acc,
        'top5_accuracy': test_top5_acc,
        'precision_macro': test_precision_macro,
        'precision_weighted': test_precision_weighted,
        'recall_macro': test_recall_macro,
        'recall_weighted': test_recall_weighted,
        'f1_macro': test_f1_macro,
        'f1_weighted': test_f1_weighted,
        'loss': test_loss
    }
    
    # 打印结果
    print(f"\n微调模型评估结果:")
    print(f"- 损失: {test_loss:.4f}")
    print(f"- Top-1准确率: {test_acc:.4f}")
    print(f"- Top-5准确率: {test_top5_acc:.4f}")
    print(f"- 精确率: {test_precision_macro:.4f} (macro)")
    print(f"- 召回率: {test_recall_macro:.4f} (macro)")
    print(f"- F1分数: {test_f1_macro:.4f} (macro)")
    
    # 保存评估指标
    save_evaluation_results(metrics, model_type)
    
    # 绘制ROC曲线
    class_names = None
    if hasattr(test_loader.dataset, 'classes'):
        class_names = test_loader.dataset.classes
    
    if len(set(all_labels)) > 1:  # 确保有多个类别
        auc_values = plot_roc_curve(all_labels, all_probs, model_type, class_names=class_names)
    
    return metrics

def main():
    """
    完整的迁移学习主流程
    """
    # 解析命令行参数
    args = parse_args()
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 打印配置信息
    print("\n==== 微调配置 ====")
    print(f"模型类型: {args.model_type}")
    print(f"预训练路径: {args.pretrained_path}")
    print(f"数据集路径: {args.data_path}")
    print(f"类别数量: {args.num_classes}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"是否冻结层: {args.freeze_layers}")
    print(f"学习率调度器: {args.lr_scheduler}")
    print(f"保存目录: {args.save_dir}")
    print("=================\n")
    
    # 检查预训练模型路径
    if args.pretrained_path is None or not os.path.exists(args.pretrained_path):
        print(f"错误: 预训练模型路径不存在 - {args.pretrained_path}")
        print("请提供有效的预训练模型路径使用 --pretrained_path 参数")
        return
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存本次运行的配置
    config_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = os.path.join(results_dir, f'finetune_config_{config_timestamp}.txt')
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(f"微调配置 ({config_timestamp})\n")
        f.write(f"模型类型: {args.model_type}\n")
        f.write(f"预训练路径: {args.pretrained_path}\n")
        f.write(f"数据集路径: {args.data_path}\n")
        f.write(f"类别数量: {args.num_classes}\n") 
        f.write(f"批次大小: {args.batch_size}\n")
        f.write(f"训练轮数: {args.epochs}\n")
        f.write(f"学习率: {args.lr}\n")
        f.write(f"是否冻结层: {args.freeze_layers}\n")
        f.write(f"学习率调度器: {args.lr_scheduler}\n")
    
    print(f"配置已保存至: {config_file}")
    
    try:
        # 初始化数据接口
        print("\n初始化数据加载器...")
        data_interface = DInterface(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # 获取数据加载器
        train_loader = data_interface.train_dataloader()
        val_loader = data_interface.val_dataloader()
        test_loader = data_interface.test_dataloader()
        
        # 初始化模型接口
        print("\n初始化模型...")
        model_interface = MInterface()
        
        # 加载预训练模型并准备迁移学习
        print(f"\n加载预训练模型: {args.pretrained_path}")
        model = load_pretrained_model(
            model_interface=model_interface,
            model_type=args.model_type,
            pretrained_path=args.pretrained_path,
            num_classes=args.num_classes
        )
        
        # 冻结模型早期层
        if args.freeze_layers:
            print("\n冻结模型早期层...")
            # 根据模型类型选择冻结策略
            if 'efficientnet' in args.model_type:
                # 冻结特征提取器，只训练分类器
                model = freeze_layers(model, freeze_names=['features'])
            elif 'resnet' in args.model_type:
                # 冻结前面几个层
                model = freeze_layers(model, freeze_until='layer3')
            elif 'vgg' in args.model_type:
                # 冻结特征提取器，只训练分类器
                model = freeze_layers(model, freeze_names=['features'])
            elif 'mobilenet' in args.model_type:
                model = freeze_layers(model, freeze_names=['features'])
            else:
                # 默认冻结策略
                print("使用默认冻结策略...")
                # 冻结前70%的层
                layers = list(model.named_parameters())
                freeze_index = int(len(layers) * 0.7)
                freeze_names = [name for name, _ in layers[:freeze_index]]
                model = freeze_layers(model, freeze_names=freeze_names)
        
        # 将模型移至设备
        model = model.to(device)
        
        # 创建损失函数
        criterion = nn.CrossEntropyLoss()
        
        # 创建优化器 - 对于微调，使用较小的学习率
        print(f"\n配置优化器，学习率: {args.lr}")
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # 创建学习率调度器
        print(f"配置学习率调度器: {args.lr_scheduler}")
        if args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.min_lr
            )
        elif args.lr_scheduler == 'reduce_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, min_lr=args.min_lr
            )
        elif args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        elif args.lr_scheduler == 'multi_step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[30, 60, 90], gamma=0.1
            )
        
        # 开始微调训练
        print("\n开始迁移学习训练...")
        model, history = transfer_learning(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=args.epochs,
            save_dir=args.save_dir
        )
        
        # 可视化训练历史
        print("\n可视化训练历史...")
        visualize_training_history(history, args.model_type)
        
        # 保存训练历史到CSV
        print("\n保存训练历史数据...")
        save_training_history_to_csv(history, args.model_type, save_dir=results_dir)
        
        # 在测试集上评估模型
        print("\n在测试集上评估微调模型...")
        metrics = evaluate_finetuned_model(model, test_loader, criterion, device, args.model_type)
        
        # 计算总运行时间
        end_time = datetime.now()
        run_time = end_time - start_time
        
        # 打印总结
        print("\n==== 微调完成 ====")
        print(f"模型类型: {args.model_type}")
        print(f"总运行时间: {run_time}")
        print(f"最终测试准确率: {metrics['accuracy']:.4f}")
        print(f"最终测试F1分数: {metrics['f1_macro']:.4f} (macro)")
        print("=================\n")
        
        # 将运行结果添加到配置文件
        with open(config_file, 'a', encoding='utf-8') as f:
            f.write(f"\n运行结果:\n")
            f.write(f"总运行时间: {run_time}\n")
            f.write(f"最终测试准确率: {metrics['accuracy']:.4f}\n")
            f.write(f"最终测试F1分数: {metrics['f1_macro']:.4f} (macro)\n")
        
        return model, history, metrics
        
    except Exception as e:
        print(f"\n错误: 迁移学习过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        
        # 记录错误
        with open(os.path.join(results_dir, f'error_log_{config_timestamp}.txt'), 'w', encoding='utf-8') as f:
            f.write(f"错误时间: {datetime.now()}\n")
            f.write(f"错误信息: {str(e)}\n")
            f.write(traceback.format_exc())
        
        return None, None, None

# 更新主入口点
if __name__ == '__main__':
    main()
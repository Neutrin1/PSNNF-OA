#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2025/03/27 14:12:53
@Author  :   Neutrin 
'''

# here put the import lib
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import argparse
import random
import datetime
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn as nn
# 设置matplotlib中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决保存图像时负号'-'显示为方块的问题
# 导入您的模型接口
from model.model_interface import MInterface


def parse_args():
    parser = argparse.ArgumentParser(description='测试模型性能')
    parser.add_argument('--checkpoint', type=str, default='D:\\Workplace\PreTraining\\result\\efficientnet-b0_epoch200_bs32\\effb0-final_model.pth', 
                        help='模型检查点路径')
    parser.add_argument('--data_dir', type=str,  default='D:\\Dataset\\mini-imagenet\\Mini-ImageNet-Dataset\\test',
                        help='测试数据目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='测试批量大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--model_type', type=str, default='efficientnet-b0',
                        help='模型类型')
    
    return parser.parse_args()


class TestDataset(Dataset):
        def __init__(self, data_dir, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            
            # 获取所有类别文件夹
            self.classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            # 获取所有图像路径及其标签
            self.samples = []
            for class_name in self.classes:
                class_dir = os.path.join(data_dir, class_name)
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, label

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model_interface = MInterface(model_type=args.model_type)
    model = model_interface.model
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"模型已加载: {args.checkpoint}")
    
    # 定义测试数据集和数据加载器
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    
    
    test_dataset = TestDataset(data_dir=args.data_dir, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_names = test_dataset.classes
    print(f"测试数据集大小: {len(test_dataset)} 图像")
    print(f"类别数量: {len(test_dataset.classes)}")
    
    # 测试模型
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0.0
    test_correct = 0

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        with tqdm(test_loader, desc="测试进度") as pbar:
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                # 使用混合精度加快推理
                with torch.amp.autocast(device.type, enabled=True):
                    outputs = model(images)
                
                # 计算损失
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                
                # 获取预测结果
                _, predictions = torch.max(outputs, 1)
                probs = F.softmax(outputs, dim=1)
                
                # 计算正确预测数量
                test_correct += torch.sum(predictions == labels.data)
                
                # 添加到列表
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    
    # 计算平均损失和准确率
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_correct.double() / len(test_loader.dataset)

    # 计算各项性能指标
    test_accuracy = accuracy_score(all_labels, all_preds)           # 整体准确率
    # 对于多分类，使用macro和weighted平均
    test_precision_macro = precision_score(all_labels, all_preds, average='macro')
    test_precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    test_recall_macro = recall_score(all_labels, all_preds, average='macro')
    test_recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    test_f1_macro = f1_score(all_labels, all_preds, average='macro')
    test_f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # 打印结果
    print("\n测试结果:")
    print(f"- 损失: {test_loss:.4f}")
    print(f"- Top-1准确率: {test_acc:.4f} (通过PyTorch计算)")
    print(f"- 准确率: {test_accuracy:.4f} (通过sklearn计算)")
    print(f"- 精确率: {test_precision_macro:.4f} (macro), {test_precision_weighted:.4f} (weighted)")
    print(f"- 召回率: {test_recall_macro:.4f} (macro), {test_recall_weighted:.4f} (weighted)")
    print(f"- F1分数: {test_f1_macro:.4f} (macro), {test_f1_weighted:.4f} (weighted)")
    
    # 保存所有指标到CSV
    metrics_dict = {
        'metric': ['loss', 'accuracy_pytorch', 'accuracy_sklearn', 'precision_macro', 'precision_weighted', 
                   'recall_macro', 'recall_weighted', 'f1_macro', 'f1_weighted'],
        'value': [test_loss, test_acc.item(), test_accuracy, test_precision_macro, test_precision_weighted, 
                  test_recall_macro, test_recall_weighted, test_f1_macro, test_f1_weighted]
    }
    
    # 创建时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 从checkpoint路径中提取模型名称
    model_name = os.path.basename(args.checkpoint).split('.')[0]
    
    # 保存指标到TXT文件
    metrics_path = os.path.join('results', f'{model_name}_{timestamp}_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"测试结果 - {timestamp}\n")
        f.write(f"模型: {args.model_type}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        for metric, value in zip(metrics_dict['metric'], metrics_dict['value']):
            f.write(f"{metric}: {value:.6f}\n")
    
    print(f"测试指标已保存至: {metrics_path}")
    # 详细的分类报告
    print("\n分类报告:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'混淆矩阵 (准确率: {test_accuracy:.4f})')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存到 confusion_matrix.png")
    
    # 将结果保存为CSV
    results_df = pd.DataFrame({
        'prediction': all_preds,
        'true_label': all_labels,
        'accuracy': (np.array(all_preds) == np.array(all_labels)).astype(int)
    })
    results_df.to_csv('test_results.csv', index=False)
    print(f"详细结果已保存到 test_results.csv")


    # 可视化预测概率分布
    plt.figure(figsize=(12, 6))
    for i, class_name in enumerate(class_names):
        probs = np.array(all_probs)[np.array(all_labels) == i]
        plt.hist(probs[:, i], bins=50, alpha=0.5, label=class_name, density=True)
    plt.title('预测概率分布')
    
    # 可视化预测结果
    # 随机选择6张图像进行预测可视化
    print("\n随机选择6张图像进行预测可视化...")
    # 创建一个新的数据集，不应用变换，保留原始图像
    vis_dataset = TestDataset(data_dir=args.data_dir, transform=None)
    


    num_samples = min(6, len(vis_dataset))
    random_indices = random.sample(range(len(vis_dataset)), num_samples)
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(random_indices):
        # 获取原始图像和标签
        image, label = vis_dataset[idx]
        
        # 应用变换以用于预测
        trans_image = test_transforms(image)
        trans_image = trans_image.unsqueeze(0).to(device)  # 添加批次维度
        
        # 进行预测
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=True):
                output = model(trans_image)
            
        # 获取预测结果
        _, pred = torch.max(output, 1)
        pred_class = pred.item()
        prob = F.softmax(output, dim=1)[0][pred_class].item()
        
        # 获取类别名称
        true_class_name = class_names[label]
        pred_class_name = class_names[pred_class]
        
        # 设置颜色：绿色表示正确预测，红色表示错误预测
        color = 'green' if pred_class == label else 'red'
        
        # 绘制图像和预测结果
        plt.subplot(2, 3, i+1)
        plt.imshow(np.array(image))
        plt.title(f"预测: {pred_class_name}\n真实: {true_class_name}\n概率: {prob:.2f}", 
                 color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('random_predictions.png', dpi=150)
    # 使用时间戳命名保存的图片
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_filename = f'random_predictions_{timestamp}.png'
    plt.savefig(prediction_filename, dpi=150)
    print(f"随机预测结果已保存到 {prediction_filename}")
    plt.show()
    
if __name__ == "__main__":
    main()
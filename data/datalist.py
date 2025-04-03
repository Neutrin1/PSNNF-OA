#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   BrestCancerData.py
@Time    :   2025/03/28 11:00:10
@Author  :   Neutrin 
'''

# here put the import lib
import os
import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from MLclf import MLclf

# 乳腺癌数据集
# （改猫狗测试用）
class BreastCancerDataset(Dataset):
    """乳腺癌图像分类数据集"""

    def __init__(self, root_dir, subset, transform=None):
        """
        初始化乳腺癌数据集
        
        Args:
            root_dir (str): 数据集根目录
            subset (str): 子集名称 ('train', 'valid', 'test')
            transform (callable, optional): 应用于图像的转换
        """
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.samples = []
        self.classes = ["cat", "dog"] 
        self.preloaded_images = {}

        # 加载数据为(samples, label)的形式
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, subset, class_name)  
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append((img_path, class_idx))



    def __len__(self):
        """返回数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """返回图片和标签"""
        img_path, label = self.samples[idx]
        # 使用OpenCV或PIL读取图像
        image = cv2.imread(img_path)            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR，转换为RGB
        
        if self.transform:
            image = self.transform(image) 
        else:
            # 基本转换：归一化和转为tensor
            image = image / 255.0
            image = image.transpose((2, 0, 1))  # HWC to CHW
            image = torch.from_numpy(image).float()
            
        return image, label
    
    @staticmethod
    def get_class_names():
        """返回类别名称"""
        return ["cat", "dog"]
    
    def count_class_distribution(self):
        """统计数据集中各类别的样本数量"""
        class_counts = [0, 0]
        for _, label in self.samples:
            class_counts[label] += 1
        return class_counts




# mini-ImageNet数据集
class MiniImageNetDataset:
    def __init__(self, transform=None, ratio_train=0.6, ratio_val=0.2, batch_size=64, seed_value=None, shuffle=True, root_path=None):
        """
        Mini-ImageNet数据集的类包装器
        
        参数:
        transform: 图像变换函数
        ratio_train: 训练集比例
        ratio_val: 验证集比例
        batch_size: 批次大小
        seed_value: 随机种子
        shuffle: 是否打乱数据
        """

        # print("检查 Mini-ImageNet 数据集...")
        # try:
        #     # 尝试下载数据集
        #     MLclf.miniimagenet_download(Download=True)
        #     print("数据集已准备就绪！")
        # except Exception as e:
        #     print(f"警告: 下载数据集时遇到问题: {str(e)}")
        #     print("将继续尝试使用已有数据...")

        # self.batch_size = batch_size

        
        
        # 如果transform为None，使用默认的transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
        
        # 加载数据集
        self.train_dataset, self.validation_dataset, self.test_dataset = MLclf.miniimagenet_clf_dataset(
            ratio_train=ratio_train, 
            ratio_val=ratio_val, 
            seed_value=seed_value,
            shuffle=shuffle,
            transform=self.transform,
            save_clf_data=True
        )
        
        # 创建数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.validation_loader = torch.utils.data.DataLoader(
            dataset=self.validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # 获取标签映射
        self.original_labels_to_marks = MLclf.labels_to_marks['mini-imagenet']
        self.original_marks_to_labels = MLclf.marks_to_labels['mini-imagenet']
        
        # 创建从0到99的新标签映射
        self.labels_to_marks = {i: name for i, name in enumerate(self.original_labels_to_marks.values())}
        self.marks_to_labels = {name: i for i, name in self.labels_to_marks.items()}
        
    
    @staticmethod
        
    def get_loaders(self):
        """返回所有数据加载器"""
        return self.train_loader, self.validation_loader, self.test_loader
    
    def get_datasets(self):
        """返回所有数据集"""
        return self.train_dataset, self.validation_dataset, self.test_dataset
    
    def get_dataset_sizes(self):
        """返回数据集大小信息"""
        return {
            'train': len(self.train_dataset),
            'validation': len(self.validation_dataset),
            'test': len(self.test_dataset)
        }
    
    def get_num_classes(self):
        """返回类别数量"""
        return len(self.labels_to_marks)
    
    def print_class_info(self):
        """打印类别信息"""
        num_classes = self.get_num_classes()
        print(f"总类别数: {num_classes}")
        print("\n所有类别:")
        for idx, class_name in self.labels_to_marks.items():
            print(f"类别 {idx}: {class_name}")



    def visualize_samples(self, loader_type='train', num_samples=5):
        """
        可视化数据样本
        
        参数:
        loader_type: 'train', 'validation' 或 'test'
        num_samples: 要可视化的样本数量
        """
        if loader_type == 'train':
            loader = self.train_loader
        elif loader_type == 'validation':
            loader = self.validation_loader
        elif loader_type == 'test':
            loader = self.test_loader
        else:
            raise ValueError("loader_type必须是'train', 'validation'或'test'之一")
            
        # 从数据加载器中获取一批数据
        images, labels = next(iter(loader))
        
        # 将归一化的图像转换回原始格式以便可视化
        images = images * 0.5 + 0.5  # 反归一化
        
        # 创建包含子图的图形
        fig, axes = plt.subplots(1, num_samples, figsize=(12, 3))
        
        for i in range(num_samples):
            # 获取图像和标签
            img = images[i].permute(1, 2, 0).cpu().numpy()
            label_idx = labels[i].item()
            class_name = self.labels_to_marks[label_idx]
            
            # 绘制图像
            axes[i].imshow(np.clip(img, 0, 1))
            axes[i].set_title(f"类别: {class_name}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()  
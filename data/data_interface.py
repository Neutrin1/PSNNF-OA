#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_interface.py
@Time    :   2025/03/26 08:56:33
@Author  :   Neutrin 
'''

# here put the import lib
"""
    此处用作数据处理的接口，将数据转换为模型可以接受的数据
"""
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
import warnings 
warnings.filterwarnings('ignore')
import MLclf
from data.datalist import BreastCancerDataset, MiniImageNetDataset




# 数据转换和增强函数
def get_train_transforms(input_size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """获取训练数据增强转换"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def get_val_transforms(input_size=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """获取验证和测试数据转换"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])




# 预定义的转换配置
TRANSFORM_CONFIGS = {
    "default": {
        "input_size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    },
    "efficientnet": {
        "input_size": (380, 380),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}


class DInterface:
    """数据接口模块"""
    
    def __init__(
        self, 
        root_path, 
        batch_size=32, 
        num_workers=4,
        transform_config="default",
        pin_memory=True
    ):
        """
        初始化乳腺癌数据模块
        
        Args:
            root_path (str): 数据集根目录
            batch_size (int): 批次大小
            num_workers (int): 数据加载器的工作线程数
            transform_config (str): 使用的转换配置名称
            pin_memory (bool): 是否使用固定内存加速GPU训练
        """
        self.root_path =  'none'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 获取转换配置
        if transform_config in TRANSFORM_CONFIGS:
            self.config = TRANSFORM_CONFIGS[transform_config]
        else:
            self.config = TRANSFORM_CONFIGS["default"]
            
        # 根据配置创建转换
        self.train_transform = get_train_transforms(
            input_size=self.config["input_size"],
            mean=self.config["mean"],
            std=self.config["std"]
        )
        
        self.val_transform = get_val_transforms(
            input_size=self.config["input_size"],
            mean=self.config["mean"],
            std=self.config["std"]
        )
        
        # self._setup()


        dataset = MiniImageNetDataset()

        self.train_dataset = dataset.train_loader
        self.val_dataset = dataset.validation_loader
        self.test_dataset = dataset.test_loader
    # def _setup(self):
    #     """准备数据集"""
    #     # 创建数据集
    #     self.train_dataset = MiniImageNetDataset(
    #         self.root_path, 'train', self.train_transform)
        
    #     self.val_dataset = MiniImageNetDataset(
    #         self.root_path, 'val', self.val_transform)
        
    #     self.test_dataset = MiniImageNetDataset(
    #         self.root_path, 'test', self.val_transform)
    
    
    # def train_dataloader(self):
    #     """返回训练数据加载器"""
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         # num_workers=self.num_workers,
    #         # pin_memory=self.pin_memory,
    #         # drop_last=True,
    #         # persistent_workers=True  # 添加此参数
    #     )
    
    # def val_dataloader(self):
    #     """返回验证数据加载器"""
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=True,
    #         # num_workers=self.num_workers,
    #         # pin_memory=self.pin_memory,
    #         # persistent_workers=True  # 添加此参数
    #     )
    
    # def test_dataloader(self):
    #     """返回测试数据加载器"""
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         shuffle=False,
    #         # num_workers=self.num_workers,
    #         # pin_memory=self.pin_memory
    #     )
    

    def get_dataset_info(self):
        """获取数据集信息"""
        train_dist = self.train_dataset.count_class_distribution()
        val_dist = self.val_dataset.count_class_distribution()
        test_dist = self.test_dataset.count_class_distribution()
        
        class_names = BreastCancerDataset.get_class_names()
        
        info = {
            "dataset_sizes": {
                "train": len(self.train_dataset),
                "val": len(self.val_dataset),
                "test": len(self.test_dataset),
            },
            "class_distribution": {
                "train": {class_names[i]: train_dist[i] for i in range(len(class_names))},
                "val": {class_names[i]: val_dist[i] for i in range(len(class_names))},
                "test": {class_names[i]: test_dist[i] for i in range(len(class_names))},
            }
        }
        return info



# 简单使用示例
if __name__ == "__main__":
    # 设置数据集路径
    root_path = 'NULL'
    # 创建数据接口实例
    data_interface = DInterface(
        root_path=root_path,
        batch_size=32,
        num_workers=4
    )
    

    if root_path != 'NULL':
        # 检查数据集路径是否存在
        # 获取数据集信息
        dataset_info = data_interface.get_dataset_info()
        print("数据集信息:")
        print(f"- 训练集大小: {dataset_info['dataset_sizes']['train']}")
        print(f"- 验证集大小: {dataset_info['dataset_sizes']['val']}")
        print(f"- 测试集大小: {dataset_info['dataset_sizes']['test']}")
        
        print("\n类别分布:")
        print(f"- 训练集: {dataset_info['class_distribution']['train']}")
        print(f"- 验证集: {dataset_info['class_distribution']['val']}")
        print(f"- 测试集: {dataset_info['class_distribution']['test']}")
        
        # 获取一个批次的数据示例
        train_loader = data_interface.train_dataloader()
        images, labels = next(iter(train_loader))
        print(f"\n批次数据形状: {images.shape}, 标签形状: {labels.shape}")
        # 可视化一些样本

        
        # 设置中文字体支持
        try:
            # 适用于Windows系统
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        except:
            # 如果不成功，尝试其他方法
            matplotlib.rcParams['font.family'] = ['Adobe Heiti Std', 'Arial Unicode MS', 'Microsoft YaHei', 'Hiragino Sans GB']
        
        def imshow(img):
            """显示图像"""
            # 反归一化
            img = img.numpy().transpose((1, 2, 0))
            mean = np.array(data_interface.config["mean"])
            std = np.array(data_interface.config["std"])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            plt.imshow(img)
        
        # 显示部分训练样本，确保显示两种类别各至少一张
        plt.figure(figsize=(15, 8))
        
        # 找出每个类别的样本索引
        class0_indices = [i for i, (_, label) in enumerate(data_interface.train_dataset.samples) if label == 0]
        class1_indices = [i for i, (_, label) in enumerate(data_interface.train_dataset.samples) if label == 1]
        
        # 确保有足够的样本
        if class0_indices and class1_indices:
            # 随机选择两张类别0图片和三张类别1图片（或反之）
            selected_indices = (
                np.random.choice(class0_indices, min(2, len(class0_indices)), replace=False).tolist() + 
                np.random.choice(class1_indices, min(3, len(class1_indices)), replace=False).tolist()
            )
            # 如果样本不足5张，补充随机样本
            if len(selected_indices) < 5:
                all_indices = list(range(len(data_interface.train_dataset)))
                remaining = [i for i in all_indices if i not in selected_indices]
                selected_indices.extend(np.random.choice(remaining, 5 - len(selected_indices), replace=False))
        else:
            # 如果某个类别没有样本，就随机选择
            selected_indices = np.random.choice(len(data_interface.train_dataset), 5, replace=False)
        
        # 打乱顺序
        np.random.shuffle(selected_indices)
        
        # 显示选择的样本
        for i, idx in enumerate(selected_indices[:5]):
            img, label = data_interface.train_dataset[idx]
            plt.subplot(1, 5, i+1)
            imshow(img)
            plt.title(f"类别: {BreastCancerDataset.get_class_names()[label]}", fontsize=12)
        
        plt.tight_layout()
        plt.show()
        

        
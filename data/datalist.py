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
        self.preloaded_images = {}
        self.classes = self.get_class_names(root_dir)  # 获取类别名称
        
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
    def get_class_names(root_dir=""):
        """获取数据集中的类别名称"""
        train_dir = os.path.join(root_dir, "train")

        print(f"正在查找训练目录: {train_dir}")
        # 返回训练文件夹中的所有子目录名称
        return [d for d in os.listdir(train_dir) 
                if os.path.isdir(os.path.join(train_dir, d))]
    
    def count_class_distribution(self):
        """统计数据集中各类别的样本数量"""
        class_counts = [0] * len(self.classes)  # 根据类别数量初始化
        for _, label in self.samples:
            class_counts[label] += 1
        return class_counts


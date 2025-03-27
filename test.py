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
import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
# 导入您的模型接口
from model.model_interface import MInterface


# Define dataset for test images
class DogsVsCatsTestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
        self.image_files.sort(key=lambda x: int(x.split('.')[0]))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.test_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        img_id = int(self.image_files[idx].split('.')[0])
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_id

# Define transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 创建模型接口实例
model_interface = MInterface(
    in_channels=3,           # RGB图像为3通道
    num_classes=2,           # 二分类问题
    dropout_rate=0.5,
    use_wavelet=False,       # 根据您训练时的配置设置
    device=device            # 使用当前设备
)

# 加载训练好的权重
model_path = os.path.join("checkpoints", "final_model.pth")
model_interface.load_model(model_path)

# 获取模型对象
model = model_interface.model
model.eval()  # 切换到评估模式

# Prepare test data
test_dir = r"E:\Dataset\dogs-vs-cats-redux-kernels-edition\test"
test_dataset = DogsVsCatsTestDataset(test_dir, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Make predictions
results = []
with torch.no_grad():
    for images, img_ids in tqdm.tqdm(test_loader):
        images = images.to(device)
        # outputs = model(images)
        # 使用softmax获取概率
        # probs = F.softmax(outputs, dim=1).cpu().numpy()
        probs = model(images).cpu().numpy()
        for img_id, prob in zip(img_ids.numpy(), probs):
            # 获取"狗"的概率 (索引1)
            dog_prob = prob[1]  # 第二个元素 (索引1) 是"狗"的概率
            # 转换为离散类别: 如果概率>0.5，则为1(狗)，否则为0(猫)
            prediction = 1 if dog_prob > 0.5 else 0
            results.append((img_id, prediction))

# Sort results by image ID
results.sort(key=lambda x: x[0])

# Create submission dataframe
submission_df = pd.DataFrame(results, columns=['id', 'label'])
submission_path = r"E:\Dataset\dogs-vs-cats-redux-kernels-edition\sample_submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Predictions saved to {submission_path}")
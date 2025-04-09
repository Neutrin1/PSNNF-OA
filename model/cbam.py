#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cbam.py
@Time    :   2025/04/09 15:30:20
@Author  :   Neutrin 
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """通道注意力模块
    
    参数:
        in_planes (int): 输入特征图的通道数
        ratio (int): 通道维度的缩减比例
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力模块
    
    参数:
        kernel_size (int): 卷积核大小，必须为3或7
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), '空间注意力卷积核大小必须为3或7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 连接通道注意力图
        x = torch.cat([avg_out, max_out], dim=1)
        # 应用卷积和sigmoid获得空间注意力图
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """CBAM注意力模块：结合通道注意力和空间注意力
    
    参考文献:
        [1] CBAM: Convolutional Block Attention Module (https://arxiv.org/abs/1807.06521)
    
    参数:
        in_planes (int): 输入特征图的通道数
        ratio (int): 通道注意力中的缩减比例
        kernel_size (int): 空间注意力中的卷积核大小
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先应用通道注意力
        x = x * self.ca(x)
        # 再应用空间注意力
        x = x * self.sa(x)
        return x


# 测试代码
if __name__ == "__main__":
    # 创建一个随机输入张量
    input_tensor = torch.randn(1, 64, 32, 32)
    
    # 创建CBAM模块
    cbam = CBAM(in_planes=64, ratio=8, kernel_size=7)
    
    # 前向传播
    output = cbam(input_tensor)
    
    # 打印输入输出形状
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
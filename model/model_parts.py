#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_parts.py
@Time    :   2025/03/28 16:30:08
@Author  :   Neutrin 
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
import pywt


#定义基础模块
class BasicBlock(nn.Module):
    """基础卷积块，包含卷积、批归一化和激活函数"""
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, use_relu: bool = True,
                 use_bn: bool = True, groups: int = 1,
                 activation_type: str = 'relu',
                 pool_type: Optional[str] = None,
                 pool_size: int = 2,
                 use_fc: bool = False,
                 fc_out_features: Optional[int] = None):
        super().__init__()
        self.conv = nn.Conv2d(                  
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=not use_bn  # 如果使用BN层，则不需要bias
        )                                    
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        # 多种激活函数选择
        if activation_type == 'relu':
            self.activation = nn.ReLU(inplace=True) if use_relu else nn.Identity()
        elif activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_type == 'tanh':
            self.activation = nn.Tanh()
        elif activation_type == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()
            
        # 池化层选择
        self.has_pool = pool_type is not None
        if self.has_pool:
            if pool_type == 'max':
                self.pool = nn.MaxPool2d(pool_size)
            elif pool_type == 'avg':
                self.pool = nn.AvgPool2d(pool_size)
            else:
                self.has_pool = False
                
        # 全连接层选择
        self.has_fc = use_fc and fc_out_features is not None
        if self.has_fc:
            self.fc = nn.Linear(out_channels, fc_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        
        if self.has_pool:
            x = self.pool(x)
            
        if self.has_fc:
            # 需要重塑以适应全连接层
            batch_size = x.size(0)
            x = x.view(batch_size, -1)
            x = self.fc(x)
            
        return x



class WaveletTransform(nn.Module):
    """小波变换层，用于多维特征提取"""
    def __init__(self, wavelet='db1', mode='symmetric', level=1):
        """
        参数:
            wavelet: 小波函数类型
            mode: 边界填充模式
            level: 分解级别
        """
        super().__init__()
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
    
    def forward(self, x):
        """
        对输入图像执行小波变换
        参数:
            x: 输入张量 [batch_size, channels, height, width]
        返回:
            变换后的特征 [batch_size, channels*4, height//2, width//2]
        """
        # 移动到CPU进行小波变换
        device = x.device
        x_np = x.detach().cpu().numpy()
        batch_size, channels, height, width = x_np.shape
        
        # 初始化输出数组
        output = np.zeros((batch_size, channels*4, height//2, width//2), dtype=np.float32)
        
        # 对每个样本的每个通道进行小波变换
        for batch_idx in range(batch_size):
            for channel_idx in range(channels):
                # 提取单个通道图像
                img = x_np[batch_idx, channel_idx]
                
                # 进行小波变换
                coeffs = pywt.dwt2(img, self.wavelet, mode=self.mode)
                LL, (LH, HL, HH) = coeffs
                
                # 存储结果
                output_channel_offset = channel_idx * 4
                output[batch_idx, output_channel_offset] = LL
                output[batch_idx, output_channel_offset + 1] = LH
                output[batch_idx, output_channel_offset + 2] = HL
                output[batch_idx, output_channel_offset + 3] = HH
        
        # 转换回PyTorch张量并返回
        output_tensor = torch.from_numpy(output).to(device)
        return output_tensor
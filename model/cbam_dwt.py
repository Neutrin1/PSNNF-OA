#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   cbam_dwt.py
@Time    :   2025/04/09 15:30:20
@Author  :   Neutrin 
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
# 导入小波模块
from .waveletpro import Downsamplewave,Downsamplewave1

# 小波通道注意力机制
class Waveletatt(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution

        # self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)
        # self.low_dim = nn.Conv2d(4 * in_planes, in_planes,kernel_size=3, stride=1,padding=1)
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave1(wavename=wavename)])
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        # self.ac = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        xori = x
        B, C, H, W= x.shape
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 2, 1)
        # x0,x1,x2,x3 = Downsamplewave(x)
        ##x0,x1,x2,x3= self.downsamplewavelet(x)
        y = self.downsamplewavelet(x)
        y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
        y = xori * y.expand_as(xori)       
        return y

# 小波空间注意力机制
class Waveletattspace(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution=224, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.input_resolution = input_resolution

        # self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(4 * dim)
        # self.low_dim = nn.Conv2d(4 * in_planes, in_planes,kernel_size=3, stride=1,padding=1)
        self.downsamplewavelet = nn.Sequential(*[nn.Upsample(scale_factor=2),Downsamplewave(wavename=wavename)])
        # self.downsamplewavelet = Downsamplewave(wavename=wavename)
        # self.conv1 = nn.Conv2d()
        # self.ac = nn.Sigmoid()
        # self.bn = nn.BatchNorm2d(in_planes)
        self.fc = nn.Sequential(
            # nn.Linear(in_planes, in_planes // 2, bias=False),
            # nn.ReLU(inplace=True),
            # nn.Linear(in_planes // 2, in_planes, bias=False),
            nn.Conv2d(in_planes*2, in_planes//2, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes//2, in_planes,kernel_size=1,padding= 0),
            nn.Sigmoid()
        )

        def forward(self, x):
            """
            x: B, H*W, C
            """
            xori = x
            B, C, H, W= x.shape
            # H, W = self.input_resolution
            # B, L, C = x.shape
            # assert L == H * W, "input feature has wrong size"
            # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
            x = x.view(B, H, W, C)
            x = x.permute(0, 3, 2, 1)        
            y = self.downsamplewavelet(x)
            y = self.fc(y) # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
            # y = self.fc(y).view(B, C, 1, 1)  # torch.Size([64, 256])-->torch.Size([64, 256, 1, 1])
            y = xori * y.expand_as(xori)       
            return y
        
        
class CBAM_Wavelet(nn.Module):
    """基于小波变换的CBAM注意力模块：结合小波通道注意力和小波空间注意力
    
    参数:
        in_planes (int): 输入特征图的通道数
        input_resolution (int): 输入特征图的分辨率
    """
    def __init__(self, in_planes, input_resolution=224):
        super(CBAM_Wavelet, self).__init__()
        # 使用小波通道注意力替换标准通道注意力
        self.ca = Waveletatt(input_resolution=input_resolution, in_planes=in_planes)
        
        # 使用小波空间注意力替换标准空间注意力
        self.sa = Waveletattspace(input_resolution=input_resolution, in_planes=in_planes)

    def forward(self, x):
        # 先应用小波通道注意力
        x = self.ca(x)
        # 再应用小波空间注意力
        x = self.sa(x)
        return x



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
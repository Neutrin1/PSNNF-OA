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

class DWT_Transform(nn.Module):
    """离散小波变换模块
    
    使用PyWavelets进行小波变换，将输入张量分解为四个子带：LL, LH, HL, HH
    """
    def __init__(self, wave='haar'):
        super(DWT_Transform, self).__init__()
        self.wave = wave
        
    def forward(self, x):
        # 输入x形状: [B, C, H, W]
        # 对每个样本和通道进行小波变换
        batch_size, channels, height, width = x.shape
        
        # 检查高度和宽度是否为偶数（小波变换的要求）
        if height % 2 == 1 or width % 2 == 1:
            # 如果不是偶数，进行填充
            pad_h = 0 if height % 2 == 0 else 1
            pad_w = 0 if width % 2 == 0 else 1
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height = height + pad_h
            width = width + pad_w
        
        # 初始化输出张量
        LL = torch.zeros((batch_size, channels, height//2, width//2), device=x.device)
        LH = torch.zeros((batch_size, channels, height//2, width//2), device=x.device)
        HL = torch.zeros((batch_size, channels, height//2, width//2), device=x.device)
        HH = torch.zeros((batch_size, channels, height//2, width//2), device=x.device)
        
        for b in range(batch_size):
            for c in range(channels):
                # 将张量移至CPU并转为NumPy数组
                coeffs = pywt.dwt2(x[b, c].cpu().detach().numpy(), self.wave)
                # 提取四个系数
                LL[b, c] = torch.from_numpy(coeffs[0]).to(x.device)
                (LH[b, c], HL[b, c], HH[b, c]) = [torch.from_numpy(d).to(x.device) for d in coeffs[1]]
        
        return LL, LH, HL, HH


class IDWT_Transform(nn.Module):
    """离散小波逆变换模块
    
    将四个子带系数重组为原始输入
    """
    def __init__(self, wave='haar'):
        super(IDWT_Transform, self).__init__()
        self.wave = wave
    
    def forward(self, LL, LH, HL, HH):
        # 输入形状: 每个子带 [B, C, H/2, W/2]
        batch_size, channels, h_half, w_half = LL.shape
        
        # 初始化输出张量
        recon = torch.zeros((batch_size, channels, h_half*2, w_half*2), device=LL.device)
        
        for b in range(batch_size):
            for c in range(channels):
                # 转换为NumPy并计算逆变换
                coeffs = (LL[b, c].cpu().detach().numpy(), 
                         (LH[b, c].cpu().detach().numpy(), 
                          HL[b, c].cpu().detach().numpy(), 
                          HH[b, c].cpu().detach().numpy()))
                recon[b, c] = torch.from_numpy(pywt.idwt2(coeffs, self.wave)).to(LL.device)
        
        return recon


class WaveletChannelAttention(nn.Module):
    """基于小波变换的通道注意力模块
    
    使用小波变换的低频系数(LL)和高频细节信息来生成通道注意力权重
    
    参数:
        in_planes (int): 输入特征图的通道数
        ratio (int): 通道维度的缩减比例
        wave (str): 小波类型
    """
    def __init__(self, in_planes, ratio=16, wave='haar'):
        super(WaveletChannelAttention, self).__init__()
        self.dwt = DWT_Transform(wave=wave)
        
        # 压缩MLP
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        
        # 高频特征处理
        self.high_fc = nn.Conv2d(in_planes * 3, in_planes, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 小波变换
        LL, LH, HL, HH = self.dwt(x)
        
        # 低频信息处理 (类似传统通道注意力的平均池化)
        avg_out = self.avg_pool(LL)
        avg_out = self.fc2(self.relu1(self.fc1(avg_out)))
        
        # 高频信息处理
        # 将三个高频子带合并
        high_feats = torch.cat([
            self.avg_pool(LH), 
            self.avg_pool(HL), 
            self.avg_pool(HH)
        ], dim=1)
        high_out = self.high_fc(high_feats)
        
        # 合并低频和高频信息
        out = avg_out + high_out
        return self.sigmoid(out)


class WaveletSpatialAttention(nn.Module):
    """基于小波变换的空间注意力模块
    
    使用小波变换的高频细节信息来增强空间特征
    
    参数:
        kernel_size (int): 卷积核大小，必须为3或7
        wave (str): 小波类型
    """
    def __init__(self, kernel_size=7, wave='haar'):
        super(WaveletSpatialAttention, self).__init__()
        assert kernel_size in (3, 7), '空间注意力卷积核大小必须为3或7'
        padding = 3 if kernel_size == 7 else 1
        
        self.dwt = DWT_Transform(wave=wave)
        
        # 高频信息特征整合
        self.high_conv = nn.Conv2d(3, 1, kernel_size, padding=padding, bias=False)
        
        # 低频和高频融合
        self.fusion_conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 小波变换
        LL, LH, HL, HH = self.dwt(x)
        
        # 处理低频信息
        # 上采样到原始大小
        b, c, h, w = x.shape
        LL_up = F.interpolate(torch.mean(LL, dim=1, keepdim=True), 
                             size=(h, w), mode='bilinear', align_corners=False)
        
        # 处理高频信息
        # 将三个高频子带堆叠
        high_feats = torch.cat([
            F.interpolate(torch.mean(LH, dim=1, keepdim=True), size=(h, w), 
                         mode='bilinear', align_corners=False),
            F.interpolate(torch.mean(HL, dim=1, keepdim=True), size=(h, w), 
                         mode='bilinear', align_corners=False),
            F.interpolate(torch.mean(HH, dim=1, keepdim=True), size=(h, w), 
                         mode='bilinear', align_corners=False)
        ], dim=1)
        
        high_out = self.high_conv(high_feats)
        
        # 融合低频和高频信息
        fusion = torch.cat([LL_up, high_out], dim=1)
        att_map = self.fusion_conv(fusion)
        
        return self.sigmoid(att_map)


class CBAM_DWT(nn.Module):
    """基于小波变换的CBAM注意力模块
    
    使用小波变换代替原始CBAM中的通道和空间注意力机制
    
    参数:
        in_planes (int): 输入特征图的通道数
        ratio (int): 通道注意力中的缩减比例
        kernel_size (int): 空间注意力中的卷积核大小
        wave (str): 小波类型，默认为'haar'
    """
    def __init__(self, in_planes, ratio=16, kernel_size=7, wave='haar'):
        super(CBAM_DWT, self).__init__()
        self.ca = WaveletChannelAttention(in_planes, ratio, wave)
        self.sa = WaveletSpatialAttention(kernel_size, wave)

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
    
    # 创建基于小波变换的CBAM模块
    cbam_dwt = CBAM_DWT(in_planes=64, ratio=8, kernel_size=7, wave='haar')
    
    # 前向传播
    output = cbam_dwt(input_tensor)
    
    # 打印输入输出形状
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape}")
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   CNN_model.py
@Time    :   2025/03/28 10:13:21
@Author  :   Neutrin 
'''


# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
# 基础组件
from mymodel_parts import BasicBlock, WaveletTransform



class Model(nn.Module):
    """基础深度CNN分类器"""
    def __init__(self, in_channels: int = 3, num_classes: int = 2,
                 layer_config: List[Tuple[int, int, int]] = None,
                 use_adaptive_pool: bool = True,
                 dropout_rate: float = 0.5):
        """
        参数:
            in_channels: 输入通道数
            num_classes: 分类类别数
            layer_config: 每层配置 [(out_channels, kernel_size, stride), ...]
            use_adaptive_pool: 是否使用自适应池化
            dropout_rate: Dropout率
        """
        super().__init__()
        
        # 默认配置: 4个卷积层
        if layer_config is None:
            layer_config = [
                (64, 3, 1),   # out_channels, kernel_size, stride
                (128, 3, 2),
                (256, 3, 2),
                (512, 3, 2)
            ]
        
        # 构建卷积层
        layers = []
        in_ch = in_channels
        for out_ch, kernel_size, stride in layer_config:
            layers.append(
                BasicBlock(in_ch, out_ch, kernel_size, stride, padding=kernel_size//2)
            )
            in_ch = out_ch
        
        self.features = nn.Sequential(*layers)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) if use_adaptive_pool else nn.Identity()
        # 分类器
        self.dropout = nn.Dropout(dropout_rate)
        last_out_ch = layer_config[-1][0] if layer_config else in_channels
        self.classifier = nn.Linear(last_out_ch, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        # 仅在评估模式下应用 softmax
        if not self.training:
            x = F.softmax(x, dim=1)

        return x


class WaveletCNN(nn.Module):
    """融合小波变换的深度CNN分类器"""
    def __init__(self, in_channels: int = 3, num_classes: int = 10,
                 layer_config: List[Tuple[int, int, int]] = None,
                 use_adaptive_pool: bool = True,
                 dropout_rate: float = 0.5,
                 wavelet_type: str = 'db1'):
        """
        参数:
            in_channels: 输入通道数
            num_classes: 分类类别数
            layer_config: 每层配置 [(out_channels, kernel_size, stride), ...]
            use_adaptive_pool: 是否使用自适应池化
            dropout_rate: Dropout率
            wavelet_type: 小波类型
        """
        super().__init__()
        
        # 小波变换层
        self.wavelet_transform = WaveletTransform(wavelet=wavelet_type)
        # 小波变换后通道数会变为原来的4倍
        wavelet_channels = in_channels * 4
        
        # 默认配置: 4个卷积层
        if layer_config is None:
            layer_config = [
                (64, 3, 1),   # out_channels, kernel_size, stride
                (128, 3, 2),
                (256, 3, 2),
                (512, 3, 2)
            ]
        
        # 构建卷积层 - 常规路径
        conv_layers = []
        in_ch = in_channels
        for out_ch, kernel_size, stride in layer_config:
            conv_layers.append(
                BasicBlock(in_ch, out_ch, kernel_size, stride, padding=kernel_size//2)
            )
            in_ch = out_ch
        
        self.conv_features = nn.Sequential(*conv_layers)
        
        # 构建卷积层 - 小波路径
        # 注意：小波变换已经将输入尺寸减半，因此第一层使用stride=1
        wavelet_layers = []
        in_ch = wavelet_channels
        
        # 第一层特殊处理，stride设为1而不是原始配置中的值
        first_layer = layer_config[0]
        wavelet_layers.append(
            BasicBlock(in_ch, first_layer[0], first_layer[1], stride=1, padding=first_layer[1]//2)
        )
        in_ch = first_layer[0]
        
        # 处理剩余层
        for out_ch, kernel_size, stride in layer_config[1:]:
            wavelet_layers.append(
                BasicBlock(in_ch, out_ch, kernel_size, stride, padding=kernel_size//2)
            )
            in_ch = out_ch
        
        self.wavelet_features = nn.Sequential(*wavelet_layers)
        
        # 特征融合层
        last_out_ch = layer_config[-1][0]
        self.fusion = nn.Conv2d(last_out_ch * 2, last_out_ch, kernel_size=1)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) if use_adaptive_pool else nn.Identity()
        
        # 分类器
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(last_out_ch, num_classes)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 常规卷积特征
        conv_out = self.conv_features(x)
        
        # 小波变换特征
        wavelet_x = self.wavelet_transform(x)
        wavelet_out = self.wavelet_features(wavelet_x)
        
        # 检查和打印特征图尺寸（调试时可以使用）
        # print(f"常规特征尺寸: {conv_out.shape}, 小波特征尺寸: {wavelet_out.shape}")
        
        # 如果尺寸不匹配，使用自适应池化调整小波特征尺寸
        if conv_out.shape[2:] != wavelet_out.shape[2:]:
            wavelet_out = F.interpolate(
                wavelet_out, 
                size=conv_out.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # 融合特征
        fused_features = torch.cat([conv_out, wavelet_out], dim=1)
        features = self.fusion(fused_features)
        
        # 池化和分类
        features = self.adaptive_pool(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        output = self.classifier(features)
        
        if not self.training:
            output = F.softmax(output, dim=1)

        return output


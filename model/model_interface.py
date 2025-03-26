#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model_interface.py
@Time    :   2025/03/26 09:43:25
@Author  :   Neutrin 
'''

# here put the import lib
"""
model_interface.py - 基础深度卷积神经网络模块

功能：
1. 提供标准化的CNN模型实现
2. 支持配置化模型参数
3. 提供模型创建和参数统计接口
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
"""
    创建模型接口
"""

#定义基础模块
class BasicBlock(nn.Module):
    """基础卷积块，包含卷积、批归一化和激活函数"""
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, use_relu: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(                  
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )                                    # 卷积
        self.pool = nn.MaxPool2d(2, 2) if stride == 2 else nn.Identity()  # 池化
        self.bn = nn.BatchNorm2d(out_channels)              # 批归一化
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()  # 激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Model(nn.Module):
    """基础深度CNN分类器"""
    def __init__(self, in_channels: int = 3, num_classes: int = 10,
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
        return x

def MInterface(
    in_channels: int = 3,
    num_classes: int = 10,
    layer_config: Optional[List[Tuple[int, int, int]]] = None,
    use_adaptive_pool: bool = True,
    dropout_rate: float = 0.5,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, int]:
    """
    创建并初始化CNN模型
    参数:
        in_channels: 输入通道数
        num_classes: 分类类别数
        layer_config: 每层配置 [(out_channels, kernel_size, stride), ...]
        use_adaptive_pool: 是否使用自适应池化
        dropout_rate: Dropout率
        device: 模型设备
    返回:
        model: 初始化的CNN模型
        total_params: 模型参数量
    """
    model = Model(
        in_channels=in_channels,
        num_classes=num_classes,
        layer_config=layer_config,
        use_adaptive_pool=use_adaptive_pool,
        dropout_rate=dropout_rate
    )
    
    if device is not None:
        model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    return model, total_params


if __name__ == "__main__":
    # 模块测试代码
    print("测试默认配置CNN:")
    model, param_count = MInterface(num_classes=10)
    print(f"\n模型结构:")
    print(model)
    print(f"\n总参数量: {param_count:,}")
    
    print("\n测试自定义配置CNN:")
    custom_config = [
        (32, 3, 1),
        (64, 3, 2),
        (128, 3, 1),
        (256, 3, 2),
        (512, 3, 1)
    ]
    model, param_count = MInterface(
        layer_config=custom_config,
        dropout_rate=0.3
    )
    print(f"\n模型结构:")
    print(model)
    print(f"\n总参数量: {param_count:,}")
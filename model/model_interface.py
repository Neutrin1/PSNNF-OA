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
4. 支持小波变换进行多维特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
import os
import pywt
import datetime

from .mymodel import Model,WaveletCNN
# unet
from .unet import UNet
# efficientnet
from .efficientnet import EfficientNet



"""
    创建模型接口
"""

class MInterface:
    """模型接口类，用于创建、加载、保存模型"""
    
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 2, 
                 layer_config: Optional[List[Tuple[int, int, int]]] = None, 
                 dropout_rate: float = 0.5, 
                 wavelet_type: str = 'db1',
                 device: Optional[torch.device] = None,
                 # 默认使用自定义CNN
                 model_type: str = 'cnn'):
                
        """
        初始化模型接口
        
        Args:
            in_channels: 输入通道数
            num_classes: 分类类别数
            layer_config: 每层配置 [(out_channels, kernel_size, stride), ...]
            dropout_rate: Dropout率
            use_wavelet: 是否使用小波变换
            wavelet_type: 小波类型
            device: 模型设备
        """
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.layer_config = layer_config
        self.dropout_rate = dropout_rate
        self.wavelet_type = wavelet_type
        self.model_type = model_type
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 创建模型
        self.model, self.param_count = self._create_model()
        print(f"创建模型成功，参数数量: {self.param_count:,}")
        
    def _create_model(self) -> Tuple[nn.Module, int]:
        """创建模型实例"""
        if self.model_type == 'waveletcnn':
            model = WaveletCNN(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                layer_config=self.layer_config,
                dropout_rate=self.dropout_rate,
                wavelet_type=self.wavelet_type
            )
            print(f"使用小波变换CNN模型，小波类型: {self.wavelet_type}")
        elif self.model_type == 'cnn':
            model = Model(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                layer_config=self.layer_config,
                dropout_rate=self.dropout_rate
            )
            print("使用我的CNN模型")
        elif self.model_type == UNet:                               #Unet网络
            model = UNet(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
            )
            print("使用Unet模型")
        else :
            model = EfficientNet.from_name('efficientnet-b0')
            print("使用EfficientNetB0模型")
        # 移动模型到指定设备
        model = model.to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        
        return model, total_params
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向传播"""
        return self.model(x)
    
    def save_model(self, path: str) -> None:
        """保存模型权重"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到: {path}")
        
    def load_model(self, path: str) -> None:
        """加载模型权重"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"已从 {path} 加载模型")



# 独立使用的
# 创建模型的辅助函数
def create_model(
    in_channels: int = 3,
    num_classes: int = 2,
    layer_config: Optional[List[Tuple[int, int, int]]] = None,
    use_adaptive_pool: bool = True,
    dropout_rate: float = 0.5,
    model_type: str = 'cnn',
    wavelet_type: str = 'db1',
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
        use_wavelet: 是否使用小波变换
        wavelet_type: 小波类型
        device: 模型设备
    返回:
        model: 初始化的CNN模型
        total_params: 模型参数量
    """
    if model_type == 'waveletcnn':
        model = WaveletCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            layer_config=layer_config,
            use_adaptive_pool=use_adaptive_pool,
            dropout_rate=dropout_rate,
            wavelet_type=wavelet_type
        )
    elif model_type == 'cnn':
        model = Model(
            in_channels=in_channels,
            num_classes=num_classes,
            layer_config=layer_config,
            use_adaptive_pool=use_adaptive_pool,
            dropout_rate=dropout_rate
        )
    elif model_type == UNet:                               #Unet网络
            model = UNet(
                in_channels=in_channels,
                num_classes=num_classes,
        )
    
    else :
        model = EfficientNet.from_name('efficientnet-b0')

    if device is not None:
        model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    return model, total_params

# 获取默认和推荐的网络配置
def get_recommended_configs():
    """返回一些推荐的网络配置"""
    configs = {
        "small": [
            (32, 3, 1),
            (64, 3, 2),
            (128, 3, 2)
        ],
        "medium": [
            (64, 3, 1),
            (128, 3, 2),
            (256, 3, 2),
            (512, 3, 2)
        ],
        "large": [
            (64, 3, 1),
            (128, 3, 1),
            (256, 3, 2),
            (512, 3, 1),
            (512, 3, 2),
            (1024, 3, 1)
        ]
    }
    return configs
# 获取可用的小波类型
def get_available_wavelets():
    """返回可用的小波类型列表"""
    return pywt.wavelist(family=None)

if __name__ == "__main__":

    # 创建日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"model_test_results_{timestamp}.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        # 测试各种模型
        models_to_test = {
            "标准CNN": {"model_type": "cnn"},
            "小波CNN": {"model_type": "waveletcnn", "wavelet_type": "db4"},
            "UNet": {"model_type": "unet"},
            "EfficientNet": {"model_type": "efficientnet"}
        }
        
        # 测试模型创建函数
        f.write("=== 测试模型创建函数 ===\n")
        for name, params in models_to_test.items():
            model, param_count = create_model(num_classes=2, **params)
            f.write(f"{name}模型参数量: {param_count:,}\n")
        
        # 测试模型接口
        f.write("\n=== 测试模型接口 ===\n")
        for name, params in models_to_test.items():
            interface = MInterface(num_classes=2, **params)
            f.write(f"{name}接口参数量: {interface.param_count:,}\n")
        
        # 简洁打印可用的小波类型
        wavelets = get_available_wavelets()
        f.write(f"\n可用的小波类型: {len(wavelets)}种\n")
        
        # 测试推荐配置
        f.write("\n=== 测试推荐配置 ===\n")
        for name, config in get_recommended_configs().items():
            for model_type in ["cnn", "waveletcnn"]:
                model, params = create_model(layer_config=config, num_classes=2, model_type=model_type)
                f.write(f"{name}配置 - {model_type}: {params:,}参数\n")
                
        # 打印UNet详细结构
        def print_model_structure(model):
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f"可训练参数总量: {total_params:,}\n")
            
        f.write("\n=== 测试模型结构详情 ===\n")
        unet_model, _ = create_model(num_classes=2, model_type="unet")
        print_model_structure(unet_model)
        f.write(str(unet_model))

        # 测试EfficientNet模型
        f.write("\n=== 测试EfficientNet模型 ===\n")
        efficientnet_model, _ = create_model(num_classes=2, model_type="efficientnet")
        print_model_structure(efficientnet_model)
        f.write(str(efficientnet_model))
    
    print(f"模型测试结果已保存到 {log_file}")

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
from CNN_model import Model, WaveletCNN
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
            print("使用标准CNN模型")
        else:                               #Unet网络
            model = Model(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                layer_config=self.layer_config,
                dropout_rate=self.dropout_rate
            )
        
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
def create_cnn_model(
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
    else:
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
    # 模块测试代码
    print("测试默认配置CNN:")
    model, param_count = create_cnn_model(num_classes=2)
    print(f"模型参数量: {param_count:,}")
    
    # 测试小波CNN
    print("\n测试小波变换CNN:")
    model, param_count = create_cnn_model(
        num_classes=2, 
        model_type = 'waveletcnn',
        wavelet_type='db4'
    )
    print(f"小波CNN模型参数量: {param_count:,}")
    
    # 测试模型接口
    print("\n测试模型接口:")
    model_interface = MInterface(num_classes=2)
    
    # 测试小波模型接口
    print("\n测试小波变换模型接口:")
    wavelet_model_interface = MInterface(
        num_classes=2, 
        model_type='waveletcnn',
        wavelet_type='haar'
    )
    
    # 查看可用小波类型
    print("\n可用的小波类型:")
    wavelets = get_available_wavelets()
    print(f"共 {len(wavelets)} 种小波类型:")
    # 每行打印10个小波类型
    for i in range(0, len(wavelets), 10):
        print(", ".join(wavelets[i:i+10]))
    
    # 测试不同配置
    configs = get_recommended_configs()
    for name, config in configs.items():
        print(f"\n测试{name}配置:")
        model, params = create_cnn_model(layer_config=config, num_classes=2)
        print(f"标准CNN参数量: {params:,}")
        
        model_w, params_w = create_cnn_model(
            layer_config=config, 
            num_classes=2, 
            model_type='waveletcnn'
        )
        print(f"小波CNN参数量: {params_w:,}")

    def print_network_structure(model):
        """打印网络结构及参数数量"""
        print("网络结构:")
        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            total_params += params
            print(f"\t{name}: {parameter.shape} → {params:,} 参数")
        
        print(f"可训练参数总量: {total_params:,}")
        print("\n完整网络结构:")
        print(model)

    if __name__ == "__main__":
        # 添加网络结构打印测试
        print("\n=== 打印标准CNN网络结构 ===")
        model, _ = create_cnn_model(num_classes=2,model_type='cnn')
        print_network_structure(model)
        
        print("\n=== 打印小波CNN网络结构 ===")
        wavelet_model, _ = create_cnn_model(num_classes=2, model_type='waveletcnn') 
        print_network_structure(wavelet_model)
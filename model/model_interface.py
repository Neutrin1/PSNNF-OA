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

"""
    创建模型接口
"""

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
        return output


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


class MInterface:
    """模型接口类，用于创建、加载、保存模型"""
    
    def __init__(self, 
                 in_channels: int = 3, 
                 num_classes: int = 2, 
                 layer_config: Optional[List[Tuple[int, int, int]]] = None, 
                 dropout_rate: float = 0.5, 
                 use_wavelet: bool = False,
                 wavelet_type: str = 'db1',
                 device: Optional[torch.device] = None):
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
        self.use_wavelet = use_wavelet
        self.wavelet_type = wavelet_type
        
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
        if self.use_wavelet:
            model = WaveletCNN(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                layer_config=self.layer_config,
                dropout_rate=self.dropout_rate,
                wavelet_type=self.wavelet_type
            )
            print(f"使用小波变换CNN模型，小波类型: {self.wavelet_type}")
        else:
            model = Model(
                in_channels=self.in_channels,
                num_classes=self.num_classes,
                layer_config=self.layer_config,
                dropout_rate=self.dropout_rate
            )
            print("使用标准CNN模型")
        
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


# 创建模型的辅助函数
def create_cnn_model(
    in_channels: int = 3,
    num_classes: int = 10,
    layer_config: Optional[List[Tuple[int, int, int]]] = None,
    use_adaptive_pool: bool = True,
    dropout_rate: float = 0.5,
    use_wavelet: bool = False,
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
    if use_wavelet:
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
        use_wavelet=True,
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
        use_wavelet=True, 
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
            use_wavelet=True
        )
        print(f"小波CNN参数量: {params_w:,}")
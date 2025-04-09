#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   efficientnet.py
@Time    :   2025/04/07 10:51:16
@Author  :   Neutrin 
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入CBAM模块
from .cbam import CBAM
# EfficientNet参数
from .efficientnet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
    calculate_output_image_size
)


"""
EfficientNet模型的实现
"""
VALID_MODELS = (
    'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
    'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
    'efficientnet-b8',

    # Support the construction of 'efficientnet-l2' without pretrained weights
    'efficientnet-l2'
)


# EfficientNet的MBConvBlock实现
class MBConvBlock(nn.Module):
    """移动端倒置残差瓶颈块。

    参数:
        block_args (namedtuple): BlockArgs, 定义在 utils.py 中。
        global_params (namedtuple): GlobalParam, 定义在 utils.py 中。
        image_size (tuple or list): [图像高度, 图像宽度]。

    参考文献:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args                           # MBConvBlockArgs，定义在 utils.py 中
        self._bn_mom = 1 - global_params.batch_norm_momentum    # 这是一个全局参数，用于批量归一化
        self._bn_eps = global_params.batch_norm_epsilon         # 这是一个全局参数，用于批量归一化
        # self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)   # 是否使用SE模块
        self.has_se = True                      # 表征CBAM模块
        self.id_skip = block_args.id_skip                       # 是否使用跳跃连接和drop connect

        # 扩展阶段 (倒置瓶颈)
        inp = self._block_args.input_filters  # 输入通道数
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # 输出通道数
        if self._block_args.expand_ratio != 1:              # 如果扩展比率不为1，则需要扩展卷积
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- 这不会修改image_size

        # 深度卷积阶段
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups使其成为深度卷积
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)



        # CBAM替代位置
        if self.has_se:
            # Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            # num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            # self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # CBAM模块
            self.cbam = CBAM(oup, ratio=16, kernel_size=7)  # ratio: 通道注意力中的缩减比例

        # 逐点卷积阶段
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """MBConvBlock的前向函数。

        参数:
            inputs (tensor): 输入张量。
            drop_connect_rate (bool): Drop connect 比率 (浮点数, 在0和1之间)。

        返回:
            处理后的该块的输出。
        """

        # 扩展和深度卷积
        x = inputs
        if self._block_args.expand_ratio != 1:  # 如果扩展比率不为1，则需要扩展卷积
            x = self._expand_conv(inputs)       # 扩展卷积
            x = self._bn0(x)                    # 批量归一化
            x = self._swish(x)                  # Swish激活函数

        x = self._depthwise_conv(x)             # 深度卷积
        x = self._bn1(x)                        # 批量归一化
        x = self._swish(x)                      # Swish激活函数

        # CBAM模块替代位置
        if self.has_se:                         # 如果使用CBAM模块
            x = self.cbam(x)                # CBAM模块

        # 逐点卷积
        x = self._project_conv(x)
        x = self._bn2(x)

        # 跳跃连接和drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # 跳跃连接和drop connect的组合带来了随机深度。
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # 跳跃连接
        return x

    def set_swish(self, memory_efficient=True):
        """设置swish函数为内存高效版本（用于训练）或标准版本（用于导出）。

        参数:
            memory_efficient (bool): 是否使用内存高效版本的swish。
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()

# 创建了EfficientNet_CBAM模型
class EfficientNet_CBAM(nn.Module):
    """带有CBAM注意力机制的EfficientNet模型。
       最简单的加载方法是使用.from_name或.from_pretrained方法。
    
    CBAM (Convolutional Block Attention Module) 结合了通道注意力和空间注意力，
    可以增强特征表示能力。

    参数:
        blocks_args (list[namedtuple]): BlockArgs的列表，用于构建模块。
        global_params (namedtuple): 在模块间共享的GlobalParams集合。

    参考文献:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
        [2] https://arxiv.org/abs/1807.06521 (CBAM)
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        # 其余代码与原始EfficientNet相同
        # ...
        assert isinstance(blocks_args, list), 'blocks_args应该是一个列表'
        
        assert len(blocks_args) > 0, 'block args必须大于0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # 批量归一化参数
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # 根据图像尺寸获取静态或动态卷积
        image_size = global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # 主干网络
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # 输出通道数
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # 构建模块
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # 基于深度乘数更新模块的输入和输出滤波器
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # 第一个模块需要处理步长和滤波器尺寸增加
            self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_args.stride)
            if block_args.num_repeat > 1:  # 修改block_args以保持相同的输出尺寸
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_args.stride)  # stride = 1

        # 顶部
        in_channels = block_args.output_filters  # 最终模块的输出
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # 最终线性层
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # 默认设置激活函数为内存高效的swish
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """设置swish函数为内存高效版本（用于训练）或标准版本（用于导出）。

        参数:
            memory_efficient (bool): 是否使用内存高效版本的swish。
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """使用卷积层从减少层i中提取特征，其中i在[1, 2, 3, 4, 5]中。

        参数:
            inputs (tensor): 输入张量。

        返回:
            包含减少层i（i在[1, 2, 3, 4, 5]中）的中间特征的字典。
            示例:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # 主干
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # 模块
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # 缩放drop connect率
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # 顶部
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs):
        """使用卷积层提取特征。

        参数:
            inputs (tensor): 输入张量。

        返回:
            EfficientNet模型中最终卷积层的输出。
        """
        # 主干
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # 模块
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # 缩放drop connect率
            x = block(x, drop_connect_rate=drop_connect_rate)

        # 顶部
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """EfficientNet的前向函数。
           调用extract_features提取特征，应用最终线性层，并返回logits。

        参数:
            inputs (tensor): 输入张量。

        返回:
            处理后的模型输出。
        """
        # 卷积层
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
            
        x = self.extract_features(inputs) 
        # 池化和最终线性层
        x = self._avg_pooling(x)
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """根据名称创建EfficientNet模型。

        参数:
            model_name (str): EfficientNet的名称。
            in_channels (int): 输入数据的通道数。
            override_params (其他关键字参数):
                用于覆盖模型全局参数的参数。
                可选键:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        返回:
            一个EfficientNet模型。
        """
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_args, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False,
                        in_channels=3, num_classes=1000, **override_params):
        """根据名称创建一个EfficientNet模型。

        参数:
            model_name (str): EfficientNet的名称。
            weights_path (None或str):
                str: 本地磁盘上预训练权重文件的路径。
                None: 使用从互联网下载的预训练权重。
            advprop (bool):
                是否加载使用advprop训练的预训练权重
                （当weights_path为None时有效）。
            in_channels (int): 输入数据的通道数。
            num_classes (int):
                分类的类别数。
                控制最终线性层的输出大小。
            override_params (其他关键字参数):
                用于覆盖模型全局参数的参数。
                可选键:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'

        返回:
            一个预训练的EfficientNet模型。
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """获取给定EfficientNet模型的输入图像尺寸。

        参数:
            model_name (str): EfficientNet的名称。

        返回:
            输入图像尺寸（分辨率）。
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """验证模型名称。

        参数:
            model_name (str): EfficientNet的名称。

        返回:
            bool: 是否为有效名称。
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name应该是以下之一: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """如果in_channels不等于3，调整模型第一个卷积层以适应in_channels。

        参数:
            in_channels (int): 输入数据的通道数。
        """
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)

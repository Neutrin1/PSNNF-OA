#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py
@Time    :   2025/03/26 08:36:57
@Author  :   Neutrin 
'''

# here put the import lib
from .model_interface import (
    MInterface, 
    # Model, 
    # BasicBlock, 
    # WaveletCNN, 
    # WaveletTransform, 
    # create_cnn_model, 
    get_recommended_configs, 
    get_available_wavelets
)

__all__ = [
    'MInterface',
    # 'Model', 
    # 'BasicBlock',
    # 'WaveletCNN',
    # 'WaveletTransform',
    # 'create_cnn_model',
    'get_recommended_configs',
    'get_available_wavelets'
]
#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn
from torch.nn import LeakyReLU

import numpy as np

def cal_size_list(in_channels, out_channels, layer_num):
    return np.linspace(
        in_channels, out_channels,
        layer_num+1, dtype='int'
    )

def MLP(size_list, last_activation=nn.LeakyReLU, activation=nn.LeakyReLU,
        last_bias=True, bias=True):
    last_bias = bias and last_bias
    return nn.Sequential(
        *(
            nn.Sequential(nn.Linear(size_list[ln], size_list[ln+1], bias=(bias if ln != len(size_list)-2 else last_bias)),
                           activation() if ln != len(size_list)-2 else last_activation())
            for ln in range(len(size_list)-1)
        )
    )

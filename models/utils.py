#!/usr/bin/env python
# coding=utf-8

import torch
from torch.nn import LeakyReLU
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as gnn_softmax

def get_activation_func(name):
    return globals()[name]

def homo_gnn_softmax(x, index, size=None):
    '''
    re-scale so that homo_softmax(s*x) = homo_softmax(x) when s > 0
    '''
    assert(not torch.sum(torch.isnan(x)))
    x_max = global_max_pool(x, index, size=size)
    assert(not torch.sum(torch.isnan(x_max)))
    x_min = -global_max_pool(-x, index, size=size)
    assert(not torch.sum(torch.isnan(x_min)))
    x_diff = (x_max-x_min)[index]
    assert(not torch.sum(torch.isnan(x_diff)))
    zero_mask = (x_diff == 0).type(torch.float)
    x_diff = torch.ones_like(x_diff)*zero_mask + x_diff*(1.-zero_mask)
    x = x/x_diff
    assert(not torch.sum(torch.isnan(x)))
    return gnn_softmax(x, index, size)

#========Testing==============
import numpy as np
from torch_geometric.nn import global_add_pool

def test_homo_gnn_softmax():
    for _ in range(1000):
        dim = np.random.randint(2,100)
        batch_size = np.random.randint(1,dim+1)

        x = torch.rand([dim])
        index = torch.randint(batch_size, [dim])
        size = batch_size
        s = np.random.rand()*1000
        while abs(s-1) < 1e-3:
            s = np.random.rand()*1000

        y = gnn_softmax(x, index, size)
        homo_y = homo_gnn_softmax(x, index, size)
        sy = gnn_softmax(s*x, index, size)
        homo_sy = homo_gnn_softmax(s*x, index, size)

        print(dim, batch_size, y.shape, homo_y.shape, sy.shape, homo_sy.shape)

        assert(not torch.sum(torch.isnan(y)))
        assert(not torch.sum(torch.isnan(homo_y)))
        assert(not torch.sum(torch.isnan(sy)))
        assert(not torch.sum(torch.isnan(homo_sy)))

        assert(torch.sum((homo_y-homo_sy)**2) < 1e-2)
        # if torch.max(global_add_pool(torch.zeros_like(x)+1, index, size)) > 1:
            # assert(torch.sum((y-sy)**2) > 1e-2)



#!/usr/bin/env python
# coding=utf-8

import torch
import torch.nn as nn

from torch_geometric.utils import softmax as gnn_softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter_add

from .utils import homo_gnn_softmax
from .classical_layers import cal_size_list, MLP

class _Aggregation(nn.Module):
    def __init__(self,*args,**kwargs):
        super(_Aggregation,self).__init__()
        self.homogeneous_flag = True
    def forward(self, *args, **kwargs):
        raise NotImplementedError

class MaxAggregation(_Aggregation):
    def forward(self, x, index, size, *args, **kwargs):
        return global_max_pool(x, index, size=size)

class MinAggregation(_Aggregation):
    def forward(self, x, index, size, *args, **kwargs):
        return -global_max_pool(-x, index, size=size)

class MeanAggregation(_Aggregation):
    def forward(self,x, index, size, *args, **kwargs):
        return global_mean_pool(x, index, size=size)

class SumAggregation(_Aggregation):
    def forward(self,x, index, size, *args, **kwargs):
        return global_add_pool(x, index, size=size)

class IdentityAggregation(_Aggregation):
    def forward(self,x, index, size, *args, **kwargs):
        return x

class AttentionAggregation(_Aggregation):
    def __init__(self, key_dim=None, value_dim=None, query_dim=None, embedding_layer_num=0, output_dim=None, homogeneous_flag=False, *args, **kwargs):
        super(AttentionAggregation, self).__init__()
        if key_dim is None or value_dim is None or query_dim is None:
            raise ValueError('Concrete dimensionality required for key', key_dim, 'value', value_dim, 'query', query_dim)

        self.homogeneous_flag = homogeneous_flag
        score_size_list = cal_size_list(key_dim+query_dim, 1, 1)
        self.score_module = MLP(score_size_list, last_activation=nn.Identity)
        if embedding_layer_num == 0:
            self.embedding_module = nn.Identity()
        else:
            embedding_size_list = cal_size_list(value_dim, output_dim or value_dim, embedding_layer_num)
            self.embedding_module = MLP(embedding_size_list, bias=not self.homogeneous_flag)
    def forward(self, keys, values, query, index, size, *args ,**kwargs):
        '''
        Note that given the special situation of GNNs,
        the definition of query is different from classical attentions.
        Here, query is only one global feature that is shared by all graphs.
        In most cases, query = torch.zeros([1,0])
        '''
        values = self.embedding_module(values)
        scores = torch.cat([keys, query.unsqueeze(0).expand([keys.size(0),-1])], dim=-1)
        scores = self.score_module(scores)
        softmax = homo_gnn_softmax if self.homogeneous_flag else gnn_softmax
        weights = softmax(scores.squeeze(), index, size)
        return scatter_add(values*weights.unsqueeze(-1), index, dim=0, dim_size=size)

class GenAttentionAggregation(AttentionAggregation):
    def forward(self, keys, values, query, index, size, *args ,**kwargs):
        mean = super(GenAttentionAggregation, self).forward(keys, values, query, index, size, *args ,**kwargs)
        sizes = global_add_pool(torch.ones_like(keys[:,0:1]), index, size=size)
        # print(mean.shape, sizes.shape)
        return mean * sizes

class HeadTailAggregation(_Aggregation):
    def forward(self, x, index, size, *args, **kwargs):
        index_set = sorted(list(set(index.tolist())))
        head_index = torch.stack(
            [torch.min(torch.arange(index.size(0))[index==i])
            for i in index_set]
        )
        tail_index = torch.stack(
            [torch.max(torch.arange(index.size(0))[index==i])
            for i in index_set]
        )
        # print(index, head_index, tail_index, x.shape)
        x = torch.cat([x[head_index], x[tail_index]], dim=-1)
        return x

class AggregationLayers(_Aggregation):
    def __init__(self, layer_name='Max', key_dim=None, value_dim=None, query_dim=None,
                 embedding_layer_num=0, output_dim=None, homogeneous_flag=False,):
        super(AggregationLayers, self).__init__()
        assert('Attention' in layer_name or output_dim is None or output_dim==value_dim)
        self.layer_name, self.homogeneous_flag = layer_name, homogeneous_flag
        self.module = globals()['%sAggregation'%self.layer_name](
            key_dim=key_dim, value_dim=value_dim, query_dim=query_dim, embedding_layer_num=embedding_layer_num,
            output_dim=output_dim, homogeneous_flag=self.homogeneous_flag,
        )
    def forward(self, keys, values, query, index, size, ):
        if 'Attention' in self.layer_name:
            return self.module(keys, values, query, index, size, )
        else:
            return self.module(values, index, size)

class _ReadoutLayers(nn.Module):
    def __init__(self, input_feat_flag=True, x_dim=None, input_x_dim=None, output_x_dim=None,
                 layer_name='Max', homogeneous_flag=False, embedding_layer_num=0):
        super(_ReadoutLayers, self).__init__()
        self.input_feat_flag = input_feat_flag
        self.homogeneous_flag = homogeneous_flag
        self.module = AggregationLayers(key_dim=(x_dim+input_x_dim if input_feat_flag else x_dim),
                                        value_dim=x_dim, query_dim=0,
                                        layer_name=layer_name,
                                        embedding_layer_num=embedding_layer_num,
                                        output_dim=output_x_dim,
                                        homogeneous_flag=homogeneous_flag)
    def cal_keys(self, data):
        if self.input_feat_flag:
            return torch.cat([data.input_x, data.x], dim=-1)
        else:
            return data.x
    def forward(self, data):
        keys, values = self.cal_keys(data), data.x
        query = torch.zeros_like(data.x[0,0:0])
        if 'batch' in data.__dict__.keys():
            index, size = data.batch, data.num_graphs
        else:
            index, size = torch.zeros([data.x.size(0)], dtype=torch.long, device=data.x.device), 1
        return self.module(keys, values, query, index, size)

def ReadoutLayers(input_feat_flag=True, x_dim=None, input_x_dim=None, output_x_dim=None,
                  layer_name='Max', homogeneous_flag=False, embedding_layer_num=0):
    return _ReadoutLayers(input_feat_flag=input_feat_flag, x_dim=x_dim,
                            input_x_dim=input_x_dim, output_x_dim=output_x_dim,
                            layer_name = layer_name,
                            homogeneous_flag=homogeneous_flag,
                            embedding_layer_num=embedding_layer_num)


#============Testing==============

import numpy as np
import os, sys, random, time, copy
from dataset.utils import gen_edge_index
from torch_geometric.data import Data, Batch

def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def _generate_data(x_dim, input_x_dim, ):
    graph_type = random.choice(['random', 'knn', 'mesh', 'lobster'])
    node_num = np.random.randint(low=5, high=100)
    device = getDevice()

    edge_index, node_num = gen_edge_index(index_generator=graph_type, node_num=node_num, device=device)
    return Data(
        x = torch.rand([node_num, x_dim], device=device),
        input_x = torch.rand([node_num, input_x_dim], device=device),
        edge_index = edge_index,
        edge_attr = torch.rand([edge_index.size(1), 1], device=device),
    )
def _one_test_case(layer_generator):
    x_dim, input_x_dim, output_x_dim = np.random.randint(1, 100, size=3).tolist()
    output_x_dim = x_dim
    data_list = [_generate_data(x_dim, input_x_dim,) for _ in range(10)]
    data = Batch.from_data_list(data_list)
    layer = layer_generator(x_dim=x_dim, input_x_dim=input_x_dim, output_x_dim=output_x_dim)
    layer, data = layer.to(data.x.device), data.to(data.x.device)
    output_x = layer(data)

    # Test output dimensionality
    assert(output_x.size(1) == output_x_dim)

    # Test correctness
    layer_name = layer.module.layer_name
    if layer_name in ['max', 'min', 'mean', 'sum']:
        if layer_name == 'Max':
            y = torch.stack([torch.max(d.x, dim=0)[0] for d in data_list], dim=0)
        elif layer_name == 'Min':
            y = torch.stack([torch.min(d.x, dim=0)[0] for d in data_list], dim=0)
        elif layer_name == 'Mean':
            y = torch.stack([torch.mean(d.x, dim=0) for d in data_list], dim=0)
        elif layer_name == 'Sum':
            y = torch.stack([torch.sum(d.x, dim=0) for d in data_list], dim=0)
        assert(torch.max(torch.abs(y-output_x)) < 1e-4)

    # Test homogeneous
    if layer.homogeneous_flag:
        s = np.random.rand()*1000.
        node_num_max = max([d.x.size(0) for d in data_list])
        data.x, data.input_x = data.x*s, data.input_x*s
        assert(torch.max(torch.abs(output_x*s-layer(data))) < 1e-3*node_num_max)
        data.x, data.input_x = data.x/s, data.input_x/s

    # Test backward
    loss_hist = []
    mlp = MLP(cal_size_list(x_dim, x_dim, 2)).to(data.x.device)
    optimizer = torch.optim.Adam(list(layer.parameters())+list(mlp.parameters()), lr=1e-5, eps=1e-5)
    for _ in range(1000):
        optimizer.zero_grad()
        my_data = copy.deepcopy(data)
        my_data.x = mlp(data.x)
        output_x = layer(my_data)
        loss = torch.sum(output_x**2)
        loss_hist.append(loss.item())
        loss.backward()
        optimizer.step()
    corr = np.corrcoef(np.arange(len(loss_hist)), loss_hist,)[0,1]
    assert(corr < -1e-3)
def _test_layer(layer_generator):
    for _ in range(10):
        _one_test_case(layer_generator)

def test_max():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Max', **kwargs))
def test_min():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Min', **kwargs))
def test_mean():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Mean', **kwargs))
def test_sum():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Sum', **kwargs))
def test_attention():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Attention', input_feat_flag=False, homogeneous_flag=False, **kwargs))
def test_attention_input():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Attention', input_feat_flag=True, homogeneous_flag=False, **kwargs))
def test_attention_homo():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Attention', input_feat_flag=False, homogeneous_flag=True, **kwargs))
def test_attention_homo_input():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Attention', input_feat_flag=True, homogeneous_flag=True, **kwargs))
def test_attention_embed():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Attention', input_feat_flag=False, homogeneous_flag=False, embedding_layer_num=2, **kwargs))
def test_attention_input_embed():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Attention', input_feat_flag=True, homogeneous_flag=False, embedding_layer_num=2, **kwargs))
def test_attention_homo_embed():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Attention', input_feat_flag=False, homogeneous_flag=True, embedding_layer_num=2, **kwargs))
def test_attention_homo_input_embed():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='Attention', input_feat_flag=True, homogeneous_flag=True, embedding_layer_num=2, **kwargs))
def test_GenAttention():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='GenAttention', input_feat_flag=False, homogeneous_flag=False, **kwargs))
def test_GenAttention_input():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='GenAttention', input_feat_flag=True, homogeneous_flag=False, **kwargs))
def test_GenAttention_homo():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='GenAttention', input_feat_flag=False, homogeneous_flag=True, **kwargs))
def test_GenAttention_homo_input():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='GenAttention', input_feat_flag=True, homogeneous_flag=True, **kwargs))
def test_GenAttention_embed():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='GenAttention', input_feat_flag=False, homogeneous_flag=False, embedding_layer_num=2, **kwargs))
def test_GenAttention_input_embed():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='GenAttention', input_feat_flag=True, homogeneous_flag=False, embedding_layer_num=2, **kwargs))
def test_GenAttention_homo_embed():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='GenAttention', input_feat_flag=False, homogeneous_flag=True, embedding_layer_num=2, **kwargs))
def test_GenAttention_homo_input_embed():
    _test_layer(lambda **kwargs: ReadoutLayers(layer_name='GenAttention', input_feat_flag=True, homogeneous_flag=True, embedding_layer_num=2, **kwargs))

def _main():
    objects = {k:v for k,v in globals().items() if k[:5] == 'test_'}
    for k,v in objects.items():
        print()
        print('====Running %s====='%k)
        v()
if __name__ == '__main__':
    _main()

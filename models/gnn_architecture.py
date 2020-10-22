#!/usr/bin/env python
# coding=utf-8

import torch.nn as nn
from torch_geometric.data import Data, Batch

'''
MPNN* GAT* *GIN* are not substantial for homogeneous flags
Maybe because some implementation noises in pytorch-geometric
'''
class DeepGNN(nn.Module):
    def __init__(self, gnn_layer_module=None, layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None)
        super(DeepGNN, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(gnn_layer_module) for _ in range(layer_num)])
    def forward(self, data):
        kwargs = {k:v for k,v in data.__dict__.items()}
        for l in self.layers:
            data = Batch(**kwargs)
            kwargs['x'] = l(data)
            assert(not torch.sum(torch.isnan(kwargs['x'])))
        return kwargs['x'], len(self.layers)

class SharedDeepGNN(nn.Module):
    def __init__(self, gnn_layer_module=None, layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None)
        super(SharedDeepGNN, self).__init__()
        self.gnn_layer_module = gnn_layer_module
        self.layer_num = layer_num
    def forward(self, data):
        kwargs = {k:v for k,v in data.__dict__.items()}
        for _ in range(self.layer_num):
            data = Batch(**kwargs)
            kwargs['x'] = self.gnn_layer_module(data)
            assert(not torch.sum(torch.isnan(kwargs['x'])))
        return kwargs['x'], self.layer_num

# Adaptive Computational Time variants
class ACTIterGNN(nn.Module):
    def __init__(self, tao, gnn_layer_module=None, readout_module=None, confidence_module=None,
                 layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None and readout_module is not None and confidence_module is not None)
        super(ACTIterGNN, self).__init__()
        self.gnn_layer_module = gnn_layer_module
        self.readout_module = readout_module
        self.confidence_module = confidence_module
        self.layer_num = layer_num
        self.tao = tao
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio=1):
        return x + left_confidence*current_confidence*new_x
    @staticmethod
    def next_x(x, new_x, left_confidence, decreasing_ratio=1):
        return x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio=1):
        return left_confidence*(1.-current_confidence)
    @property
    def decreasing_ratio(self,):
        return None
    def forward(self, data):
        if self.layer_num == 0:
            return data.x, 0, torch.zeros_like(data.x[:,0:1])
        x, batch = data.x, data.batch
        kwargs = {k:v for k,v in data.__dict__.items()}
        kwargs.pop('x')
        new_x = x

        left_confidence = torch.ones_like(x[:,0:1])
        residual_confidence = torch.ones_like(x[:,0:1])
        zero_mask = torch.zeros_like(x[:,0:1])
        for iter_num in range(self.layer_num):
            data = Batch(x=self.next_x(x, new_x, left_confidence, self.decreasing_ratio), **kwargs)
            new_x = self.gnn_layer_module(data)
            global_feat = self.readout_module(Batch(x=new_x, **kwargs))
            current_confidence = self.confidence_module(global_feat)[batch]

            left_confidence = left_confidence - current_confidence*(1-zero_mask)
            current_zero_mask = (left_confidence < 1e-7).type(torch.float)
            residual_confidence = residual_confidence - current_confidence*(1-current_zero_mask)
            x = x + (current_confidence*(1-current_zero_mask)+residual_confidence*current_zero_mask*(1-zero_mask))*new_x
            zero_mask = current_zero_mask
            if torch.min(zero_mask).item() > 0.5:
                break;
        return x, iter_num, residual_confidence
def ACT0IterGNN(*args, **kwargs):
    return ACTIterGNN(0, *args, **kwargs)
def ACT1IterGNN(*args, **kwargs):
    return ACTIterGNN(0.1, *args, **kwargs)
def ACT2IterGNN(*args, **kwargs):
    return ACTIterGNN(0.01, *args, **kwargs)
def ACT3IterGNN(*args, **kwargs):
    return ACTIterGNN(0.001, *args, **kwargs)

class IterGNN(nn.Module):
    def __init__(self, gnn_layer_module=None, readout_module=None, confidence_module=None,
                 layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None and readout_module is not None and confidence_module is not None)
        super(IterGNN, self).__init__()
        self.gnn_layer_module = gnn_layer_module
        self.readout_module = readout_module
        self.confidence_module = confidence_module
        self.layer_num = layer_num
    def forward(self, data):
        if self.layer_num == 0:
            return data.x, 0
        x, batch = data.x, data.batch
        kwargs = {k:v for k,v in data.__dict__.items()}
        kwargs.pop('x')
        new_x = x

        left_confidence = torch.ones_like(x[:,0:1])
        for iter_num in range(self.layer_num):
            if torch.max(left_confidence).item() > 1e-7:
                data = Batch(x=self.next_x(x, new_x, left_confidence, self.decreasing_ratio), **kwargs)
                new_x = self.gnn_layer_module(data)
                global_feat = self.readout_module(Batch(x=new_x, **kwargs))
                current_confidence = self.confidence_module(global_feat)[batch]
                x = self.update_x(
                    x if iter_num != 0 else torch.zeros_like(x),
                    new_x, left_confidence, current_confidence, self.decreasing_ratio
                )
                left_confidence = self.update_confidence(left_confidence, current_confidence, self.decreasing_ratio)
            else:
                break

        return x, iter_num
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio=1):
        return x + left_confidence*current_confidence*new_x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio=1):
        return left_confidence*(1.-current_confidence)
    @property
    def decreasing_ratio(self,):
        return None
    @staticmethod
    def next_x(x, new_x, left_confidence, decreasing_ratio=1):
        return new_x
class IterNodeGNN(nn.Module):
    def __init__(self, gnn_layer_module=None, readout_module=None, confidence_module=None,
                 layer_num=1, *args, **kwargs):
        assert(gnn_layer_module is not None and readout_module is not None and confidence_module is not None)
        super(IterNodeGNN, self).__init__()
        self.gnn_layer_module = gnn_layer_module
        self.readout_module = readout_module
        self.confidence_module = confidence_module
        self.layer_num = layer_num
    def forward(self, data):
        if self.layer_num == 0:
            return data.x, 0
        x, batch = data.x, data.batch
        kwargs = {k:v for k,v in data.__dict__.items()}
        kwargs.pop('x')
        new_x = x

        left_confidence = torch.ones_like(x[:,0:1])
        for iter_num in range(self.layer_num):
            if torch.max(left_confidence).item() > 1e-7:
                data = Batch(x=self.next_x(x, new_x, left_confidence, self.decreasing_ratio), **kwargs)
                new_x = self.gnn_layer_module(data)
                # global_feat = self.readout_module(Batch(x=new_x, **kwargs))
                # current_confidence = self.confidence_module(global_feat)[batch]
                current_confidence = self.confidence_module(new_x)
                x = self.update_x(
                    x if iter_num != 0 else torch.zeros_like(x),
                    new_x, left_confidence, current_confidence, self.decreasing_ratio
                )
                left_confidence = self.update_confidence(left_confidence, current_confidence, self.decreasing_ratio)
            else:
                break

        return x, iter_num
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio=1):
        return x + left_confidence*current_confidence*new_x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio=1):
        return left_confidence*(1.-current_confidence)
    @property
    def decreasing_ratio(self,):
        return None
    @staticmethod
    def next_x(x, new_x, left_confidence, decreasing_ratio=1):
        return new_x
class DecIterGNN(IterGNN):
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio):
        return decreasing_ratio*x + left_confidence*current_confidence*new_x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio):
        return left_confidence*(1.-current_confidence)*decreasing_ratio
    @property
    def decreasing_ratio(self,):
        return (1-1e-4)
class DecIterNodeGNN(IterNodeGNN):
    @staticmethod
    def update_x(x, new_x, left_confidence, current_confidence, decreasing_ratio):
        return decreasing_ratio*x + left_confidence*current_confidence*new_x
    @staticmethod
    def update_confidence(left_confidence, current_confidence, decreasing_ratio):
        return left_confidence*(1.-current_confidence)*decreasing_ratio
    @property
    def decreasing_ratio(self,):
        return (1-1e-4)

def GNNArchitectures(gnn_layer_module=None, readout_module=None, confidence_module=None,
                     layer_name='IterGNN', layer_num=1, *args, **kwargs):
    if layer_name in ['DeepGNN', 'SharedDeepGNN']:
        return globals()[layer_name](gnn_layer_module, layer_num=layer_num, *args, **kwargs)
    elif 'Iter' in layer_name:
        return globals()[layer_name](gnn_layer_module, readout_module, confidence_module,
                                     layer_num=layer_num, *args, **kwargs)
    else:
        raise NotImplementedError('There is no GNN architecture named %s'%layer_name)


#=======Testing Architecture==========

import torch, random, os, sys, time, copy
from .gnn_layers import GNNLayers
from .gnn_aggregation import ReadoutLayers
from .classical_layers import cal_size_list, MLP
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(os.curdir))))
from dataset.utils import gen_edge_index
# from torch_geometric.data import Data, Batch
import numpy as np

def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def _generate_data(x_dim, input_x_dim, edge_attr_dim, output_x_dim, ):
    graph_type = random.choice(['random', 'knn', 'mesh', 'lobster'])
    node_num = np.random.randint(low=5, high=100)
    device = getDevice()

    edge_index, node_num = gen_edge_index(index_generator=graph_type, node_num=node_num, device=device)
    return Data(
        x = torch.rand([node_num, x_dim], device=device),
        input_x = torch.rand([node_num, input_x_dim], device=device),
        edge_index = edge_index,
        edge_attr = torch.rand([edge_index.size(1), edge_attr_dim], device=device),
    )
def _one_test_case(arch_generator):
    x_dim, input_x_dim, edge_attr_dim = np.random.randint(1, 100, size=3).tolist()
    output_x_dim, layer_num = x_dim, np.random.randint(10)
    layer_name = random.choice(['GCNConv', 'GATConv', 'GINConv', 'EpsGINConv',
                                'MPNNConv', 'MPNNMaxConv', 'PathConv', 'PathSimConv'])
    homogeneous_flag, input_feat_flag = random.choices([True, False], k=2)
    layer = GNNLayers(x_dim=x_dim, input_x_dim=input_x_dim,
                      output_x_dim=output_x_dim, edge_attr_dim=edge_attr_dim,
                      layer_name=layer_name, homogeneous_flag=homogeneous_flag,
                      input_feat_flag=input_feat_flag)
    readout_name = random.choice(['Max', 'Min', 'Mean', 'Sum', 'Attention'])
    readout_layer = ReadoutLayers(x_dim=x_dim, input_x_dim=input_x_dim,
                                  output_x_dim=output_x_dim, layer_name=readout_name,
                                  homogeneous_flag=homogeneous_flag,
                                  input_feat_flag=input_feat_flag,
                                  embedding_layer_num=0)
    confidence_layer = MLP([output_x_dim, 1], last_activation=nn.Sigmoid, bias=not homogeneous_flag)
    arch = arch_generator(gnn_layer_module=layer, readout_module=readout_layer,
                          confidence_module=confidence_layer, layer_num=layer_num,)
    print(layer_num, layer_name, homogeneous_flag, input_feat_flag, readout_name)

    data = Batch.from_data_list([_generate_data(x_dim, input_x_dim, edge_attr_dim, output_x_dim)
                                 for _ in range(10)])
    data = data.to(data.x.device)
    arch = arch.to(data.x.device)
    if 'ACT' not in arch.__class__.__name__:
        output_x, _ = arch(data)
    else:
        output_x, _, _ = arch(data)

    # Test output dimensionality
    assert(output_x.size(1) == output_x_dim)

    # Test homogeneous
    if layer.homogeneous_flag and layer_num > 0 \
            and layer_name not in ['GATConv', 'GINConv', 'EpsGINConv', 'MPNNConv', 'MPNNMaxConv']\
            and 'IterGNN' not in arch.__class__.__name__:
        s = np.random.rand()*1000.
        data.x, data.input_x, data.edge_attr = data.x*s, data.input_x*s, data.edge_attr*s
        assert(torch.max(torch.abs(output_x*s-arch(data)[0])) <= 1e-2*layer_num)
        data.x, data.input_x, data.edge_attr = data.x/s, data.input_x/s, data.edge_attr/s

    # Test backward
    if layer_num > 0:
        loss_hist = []
        optimizer = torch.optim.Adam(arch.parameters(), lr=1e-3, eps=1e-5)
        for _ in range(1000):
            optimizer.zero_grad()
            if 'ACT' not in arch.__class__.__name__:
                output_x, _ = arch(data)
            else:
                output_x, _, _ = arch(data)
            loss = torch.mean(output_x**2)
            loss_hist.append(loss.item())
            loss.backward()
            optimizer.step()
        corr = np.corrcoef(np.arange(len(loss_hist)), loss_hist,)[0,1]
        print(corr,'**',loss_hist[::100])
        assert(corr < -1e-3), corr #Assuming loss will decrease during training. But for some extremely special configurations of the models, loss may not decrease...
def _test_arch(arch_generator):
    for _ in range(10):
        _one_test_case(arch_generator)

def test_DeepGNN():
    _test_arch(lambda **kwargs: GNNArchitectures(layer_name='DeepGNN', **kwargs))
def test_SharedDeepGNN():
    _test_arch(lambda **kwargs: GNNArchitectures(layer_name='SharedDeepGNN', **kwargs))
def test_ACT0IterGNN():
    _test_arch(lambda **kwargs: GNNArchitectures(layer_name='ACT0IterGNN', **kwargs))
def test_ACT1IterGNN():
    _test_arch(lambda **kwargs: GNNArchitectures(layer_name='ACT1IterGNN', **kwargs))
def test_IterGNN():
    _test_arch(lambda **kwargs: GNNArchitectures(layer_name='IterGNN', **kwargs))
def test_UngatedIterGNN():
    _test_arch(lambda **kwargs: GNNArchitectures(layer_name='UngatedIterGNN', **kwargs))
def test_DecUngatedIterGNN():
    _test_arch(lambda **kwargs: GNNArchitectures(layer_name='DecUngatedIterGNN', **kwargs))
def test_DecTrueUngatedIterGNN():
    _test_arch(lambda **kwargs: GNNArchitectures(layer_name='DecTrueUngatedIterGNN', **kwargs))
def test_DecTrueUngatedIterNodeGNN():
    _test_arch(lambda **kwargs: GNNArchitectures(layer_name='DecTrueUngatedIterNodeGNN', **kwargs))

def _main():
    objects = {k:v for k,v in globals().items() if k[:5] == 'test_'}
    for k,v in objects.items():
        print()
        print('====Running %s====='%k)
        v()
if __name__ == '__main__':
    _main()

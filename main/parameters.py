#!/usr/bin/env python
# coding=utf-8

import sys, os, copy
import os.path as osp
curdir = osp.dirname(osp.abspath(__file__))
maindir = osp.dirname(curdir)
sys.path.append(maindir)
from dataset import *
from models import GraphGNNModels, NodeGNNModels, JKGraphGNNModels

class _Parameters(object):
    def __init__(self):
        pass
    def __str__(self,):
        return '%s: %s'%(self.__class__.__name__, str(self.__dict__))
    def __repr__(self,):
        return self.__str__()

class DatasetParam(_Parameters):
    def __init__(self, size=100000, min_num_node=4, num_num_node=30,
                 sparsity=0.5, k=8, dim=2, lobster_prob=(0.2, 0.2),
                 index_generator='random',
                 min_edge_distance=1., max_edge_distance=1.,
                 device=None, dataset_name='ShortestPathLen'):
        self.dataset_name = dataset_name
        self.size = size
        self.min_num_node = min_num_node
        self.num_num_node = num_num_node
        self.sparsity = sparsity
        self.k = k
        self.dim = dim
        self.lobster_prob = lobster_prob
        # self.connect_prob = connect_prob
        self.index_generator = index_generator
        self.min_edge_distance = min_edge_distance
        self.max_edge_distance = max_edge_distance
        self.device = device

def get_dataset_param(size=100000, min_num_node=4, num_num_node=30,
                      index_generator='random', weighted_flag=False,
                      device=None, dataset_name='ShortestPathLen'):
    param = DatasetParam(size=size,
                         min_num_node=min_num_node,
                         num_num_node=num_num_node,
                         index_generator=index_generator,
                         device=device,
                         dataset_name=dataset_name)
    if weighted_flag:
        param.min_edge_distance = 0.5
        param.max_edge_distance = 1.5
    else:
        param.min_edge_distance = param.max_edge_distance = 1.
    if index_generator == 'random':
        param.sparsity = 0.5
    elif index_generator == 'knn':
        param.k = 8
        param.dim = 1
    elif index_generator == 'mesh':
        param.dim = 2
    elif index_generator == 'lobster':
        param.lobster_prob = (0.2, 0.2)
    return param

def param2dataset(param, train_flag=True):
    other_param = copy.copy(param.__dict__)
    other_param.pop('dataset_name')
    return globals()[param.dataset_name+'Dataset'](train_flag=train_flag, **other_param)

def dataset_param2path(record_dir, dataset_param):
    weighted_flag = (dataset_param.min_edge_distance != dataset_param.max_edge_distance)
    if weighted_flag:
        standard_weighted_flag = (abs(dataset_param.min_edge_distance-0.5)<1e-5) and (abs(dataset_param.max_edge_distance-1.5)<1e-5)
    size, index_generator = dataset_param.size, dataset_param.index_generator
    min_num_node, num_num_node = dataset_param.min_num_node, dataset_param.num_num_node

    record_dir = osp.join(record_dir, dataset_param.dataset_name)
    if not osp.exists(record_dir):
        os.mkdir(record_dir)
    record_dir = osp.join(record_dir, '%s_%s'%(index_generator,(
        'weighted' if standard_weighted_flag else f'weighted_{dataset_param.min_edge_distance}_{dataset_param.max_edge_distance}'
    ) if weighted_flag else 'unweighted'))
    if not osp.exists(record_dir):
        os.mkdir(record_dir)
    record_dir = osp.join(record_dir, '%d_%d'%(min_num_node, num_num_node))
    if not osp.exists(record_dir):
        os.mkdir(record_dir)
    record_dir = osp.join(record_dir, '%s'%size)
    if not osp.exists(record_dir):
        os.mkdir(record_dir)
    return record_dir

class ModelParam(_Parameters):
    def __init__(self, embedding_layer_num=2,
                 architecture_name='IterGNN', layer_num=10,
                 module_num=1,
                 layer_name='PathConv', hidden_size=64,
                 input_feat_flag=True, homogeneous_flag=1,
                 readout_name='Max',
                 confidence_layer_num=1,
                 head_layer_num=1,
                 model_type='Graph'):
        self.embedding_layer_num = embedding_layer_num
        self.architecture_name = architecture_name
        self.layer_num = layer_num
        self.module_num = module_num
        self.layer_name = layer_name
        self.hidden_size = hidden_size
        self.input_feat_flag = input_feat_flag
        self.homogeneous_flag = homogeneous_flag
        self.readout_name = readout_name
        self.confidence_layer_num = confidence_layer_num
        self.head_layer_num = head_layer_num
        self.model_type = model_type

def get_model_param(architecture_name='IterGNN', layer_num=10,
                    module_num = 1,
                    layer_name='PathConv', homogeneous_flag=0,
                    readout_name='Max', model_type='Graph'):
    param = ModelParam(architecture_name=architecture_name,
                       layer_num=layer_num,
                       module_num=module_num,
                       layer_name=layer_name,
                       homogeneous_flag=homogeneous_flag,
                       readout_name=readout_name,
                       model_type=model_type)
    return param

def get_channels(dataset):
    if dataset.__class__.__name__ == 'DatasetParam': #for debug
        in_channel, edge_channel, out_channel = 3, 1, 1
    else: #is then the dataset
        in_channel = dataset.num_node_features
        edge_channel = dataset.num_edge_features
        out_channel = dataset.num_classes
    return in_channel, edge_channel, out_channel
def param2model(dataset, param):
    in_channel, edge_channel, out_channel = get_channels(dataset)
    model_type = param.model_type
    param_dict = {k:v for k,v in param.__dict__.items() if k != 'model_type'}
    if model_type == 'Graph':
        return GraphGNNModels(in_channel, edge_channel, out_channel, **param_dict)
    elif model_type == 'Node':
        return NodeGNNModels(in_channel, edge_channel, out_channel, **param_dict)
    elif model_type == 'JKGraph':
        return JKGraphGNNModels(in_channel, edge_channel, out_channel, **param_dict)
    else:
        raise ValueError('Wrong model_type:', model_type)

def model_param2path(record_dir, model_param):
    architecture_name, layer_num = model_param.architecture_name, model_param.layer_num
    module_num = model_param.module_num
    layer_name, homogeneous_flag = model_param.layer_name, model_param.homogeneous_flag
    readout_name = model_param.readout_name
    model_type = model_param.model_type

    record_dir = osp.join(record_dir, '%s_%d'%(architecture_name, layer_num))
    if not osp.exists(record_dir):
        os.mkdir(record_dir)

    if homogeneous_flag == 1:
        homo_name = 'homo'
    elif homogeneous_flag == 2:
        homo_name = 'shomo'
    elif homogeneous_flag == 0:
        homo_name = 'ihomo'
    record_dir = osp.join(record_dir, '%s_%s'%(layer_name, homo_name))
    if not osp.exists(record_dir):
        os.mkdir(record_dir)

    record_dir = osp.join(record_dir, readout_name)
    if not osp.exists(record_dir):
        os.mkdir(record_dir)

    record_dir = osp.join(record_dir, str(module_num))
    if not osp.exists(record_dir):
        os.mkdir(record_dir)

    record_dir = osp.join(record_dir, model_type)
    if not osp.exists(record_dir):
        os.mkdir(record_dir)
    return record_dir

class GeneralParam(_Parameters):
    def __init__(self, learning_rate=1e-3, epoch_num=1000,
                 batch_size=32, save_freq=200, log_freq=100,
                 resume_flag=False):
        self.learning_rate = learning_rate
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.resume_flag = resume_flag
        self.running_metric_name_list = ['relative_loss', 'mse_loss']

def get_general_param(learning_rate=1e-3, epoch_num=1000,
                      batch_size=32, save_freq=200,
                      log_freq=100, resume_flag=False):
    return GeneralParam(
        learning_rate=learning_rate, epoch_num=epoch_num,
        batch_size=batch_size, save_freq=save_freq,
        log_freq=log_freq, resume_flag=resume_flag,
    )

def params2path(dataset_param, model_param):
    record_dir = osp.join(maindir, 'record')
    if not osp.exists(record_dir):
        os.mkdir(record_dir)

    record_dir = dataset_param2path(record_dir, dataset_param, )
    record_dir = model_param2path(record_dir, model_param)

    logdir = osp.join(record_dir, 'logs')
    if not osp.exists(logdir):
        os.mkdir(logdir)
    print('Saving logs to ', logdir)
    return logdir

# def path2params(logdir):
    # record_dir = osp.dirname(logdir)

    # readout_name = osp.basename(record_dir)
    # record_dir = osp.dirname(record_dir)
    # layer_name, homogeneous_flag = osp.basename(record_dir).split('_')
    # homogeneous_flag = (homogeneous_flag=='homo')
    # record_dir = osp.dirname(record_dir)
    # architecture_name, layer_num = osp.basename(record_dir).split('_')
    # layer_num = int(layer_num)
    # record_dir = osp.dirname(record_dir)

    # model_param = get_model_param(
        # architecture_name=architecture_name, layer_num=layer_num,
        # layer_name=layer_name, homogeneous_flag=homogeneous_flag,
        # readout_name=readout_name,
    # )

    # size = int(osp.basename(record_dir))
    # record_dir = osp.dirname(record_dir)
    # min_num_node, num_num_node = osp.basename(record_dir).split('_')
    # min_num_node, num_num_node = int(min_num_node), int(num_num_node)
    # record_dir = osp.dirname(record_dir)
    # index_generator, weighted_flag = osp.basename(record_dir).split('_')
    # weighted_flag = (weighted_flag=='weighted')
    # record_dir = osp.dirname(record_dir)

    # dataset_param = get_dataset_param(
        # size=size, min_num_node=min_num_node, num_num_node=num_num_node,
        # index_generator=index_generator, weighted_flag=weighted_flag,
    # )

    # return dataset_param, model_param

#===========================Testing Parameters=============================

import random, torch
import numpy as np
from .utils import getDevice

def test_get_dataset_param():
    device = getDevice()
    for weighted_flag in [True, False]:
        param = get_dataset_param(index_generator='random', weighted_flag=weighted_flag, device=device)
        assert(param.sparsity == 0.5)
        assert(param.device == device)
        if weighted_flag:
            assert(param.min_edge_distance == 0.5 and param.max_edge_distance==1.5)
        else:
            assert(param.min_edge_distance == 1. and param.max_edge_distance==1.)

        param = get_dataset_param(index_generator='knn', weighted_flag=weighted_flag, device=device)
        assert(param.dim==1 and param.k==8)
        assert(param.device == device)
        if weighted_flag:
            assert(param.min_edge_distance == 0.5 and param.max_edge_distance==1.5)
        else:
            assert(param.min_edge_distance == 1. and param.max_edge_distance==1.)

        param = get_dataset_param(index_generator='mesh', weighted_flag=weighted_flag, device=device)
        assert(param.dim==2)
        assert(param.device == device)
        if weighted_flag:
            assert(param.min_edge_distance == 0.5 and param.max_edge_distance==1.5)
        else:
            assert(param.min_edge_distance == 1. and param.max_edge_distance==1.)

        param = get_dataset_param(index_generator='lobster', weighted_flag=weighted_flag, device=device)
        assert(param.lobster_prob==(0.2, 0.2))
        assert(param.device == device)
        if weighted_flag:
            assert(param.min_edge_distance == 0.5 and param.max_edge_distance==1.5)
        else:
            assert(param.min_edge_distance == 1. and param.max_edge_distance==1.)

def _test_param2dataset():
    device = getDevice()
    size = np.random.randint(1,10)
    min_num_node, num_num_node = np.random.randint(4, 100, size=2)
    weighted_flag = random.choice([True, False])
    index_generator = random.choice(['random', 'knn', 'mesh', 'lobster', ])
    param = get_dataset_param(size=size, min_num_node=min_num_node,
                              num_num_node=num_num_node, index_generator=index_generator,
                              weighted_flag=weighted_flag, device=device)
    print(param.__dict__)
    dataset = param2dataset(param)

    assert(len(dataset) == size)
    for data in dataset:
        assert(data.x.size(0) >= min_num_node)
        if 'lobster' not in index_generator:
            assert(data.x.size(0) < min_num_node+num_num_node)
        if weighted_flag:
            assert(torch.prod(data.edge_attr < 1.5))
            assert(torch.prod(data.edge_attr > 0.5))
        else:
            assert(torch.prod(data.edge_attr == 1.))
def test_param2dataset():
    for _ in range(10):
        _test_param2dataset()

def _test_param2model():
    architecture_name = random.choice(['DeepGNN', 'SharedDeepGNN', 'IterGNN'])
    layer_num = np.random.randint(1,30)
    layer_name = random.choice(['GCNConv', 'GATConv', 'GINConv', 'EpsGINConv',
                                'MPNNConv', 'MPNNMaxConv', 'PathConv', 'PathSimConv'])
    homogeneous_flag = random.choice([True, False])
    readout_name = random.choice(['Max', 'Mean', 'Sum', 'Attention'])
    param = get_model_param(
        architecture_name=architecture_name,
        layer_num=layer_num,
        layer_name=layer_name,
        homogeneous_flag=homogeneous_flag,
        readout_name=readout_name,
    )
    print(param)
    model = param2model(get_dataset_param(), param)
    assert(len(model.embedding_module) == 2)
    assert(readout_name in model.readout_module.module.module.__class__.__name__)
    assert(len(model.head_module) == 1)

    assert(architecture_name in model.gnn_module_list[0].__class__.__name__)
    print(str(model))
    assert(layer_name in str(model))
    if 'layer_num' in model.gnn_module_list[0].__dict__:
        assert(model.gnn_module_list[0].layer_num == layer_num)
    if 'confidence_module' in model.gnn_module_list[0].__dict__:
        assert(len(model.gnn_module_list[0].confidence_module) == 1)
    if 'readout_module' in model.gnn_module_list[0].__dict__:
        assert(readout_name in model.gnn_module_list[0].readout_module.module.module.__class__.__name__)

    print(model)
def test_param2model():
    for _ in range(10):
        _test_param2model()

def _main():
    objects = {k:v for k,v in globals().items() if k[:5] == 'test_'}
    for k,v in objects.items():
        print()
        print('====Running %s====='%k)
        v()
if __name__ == '__main__':
    _main()

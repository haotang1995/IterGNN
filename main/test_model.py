#!/usr/bin/env python
# coding=utf-8

from .parameters import dataset_param2path, params2path
from .parameters import param2model, param2dataset
from .evaluate import evaluate

import os.path as osp
import numpy as np
import torch, os, time, shutil, copy
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel

def test_model(model_path,
               dataset_param, model_param, test_dataset_param,
               batch_size=32, layer_num=None,
               metric_name_list=['relative_loss', 'mse_loss', 'accuracy',
                                 'post_relative_loss', 'post_mse_loss', 'post_accuracy',
                                 'layer_num',]):
    print(model_path)
    print(dataset_param, model_param, test_dataset_param)

    model_filename = osp.basename(model_path)
    new_log_dir = osp.dirname(model_path)
    log_dir_base = osp.basename(new_log_dir)
    log_dir = osp.dirname(osp.dirname(new_log_dir))
    performance_dir = osp.join(log_dir, 'performance')
    if not osp.exists(performance_dir):
        os.mkdir(performance_dir)
    performance_dir = dataset_param2path(performance_dir, test_dataset_param)
    new_performance_dir = osp.join(performance_dir, log_dir_base)
    if not osp.exists(new_performance_dir):
        os.mkdir(new_performance_dir)

    batch_size = batch_size

    parallel_flag = False
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # parallel_flag = torch.cuda.device_count() > 1
    else:
        device = torch.device('cpu')

    cur_model_path = os.path.join(new_log_dir, model_filename)
    cur_performance_dir = osp.join(new_performance_dir, model_filename)
    if not osp.exists(cur_performance_dir):
        os.mkdir(cur_performance_dir)
    if not all([osp.exists(osp.join(cur_performance_dir, ('raw_%s'%metric_name)+('' if layer_num is None else '_'+str(layer_num))+'.csv'))
                for metric_name in metric_name_list]):
        with torch.no_grad():
            test_dataset = param2dataset(test_dataset_param, train_flag=False)
            data_loader_fn = DataListLoader if parallel_flag else DataLoader
            test_data_loader = data_loader_fn(test_dataset,  batch_size)

            net = param2model(test_dataset, model_param)
            net = net.to(device)
            checkpoint = torch.load(cur_model_path, map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'])
            if layer_num is not None:
                try:
                    net.gnn_module.layer_num = layer_num
                except AttributeError:
                    for i in range(len(net.gnn_module_list)):
                        net.gnn_module_list[i].layer_num = layer_num
            if parallel_flag:
                net = DataParallel(net)

            test_metric_list, y_list = evaluate(net, test_data_loader, device,
                                                parallel_flag=parallel_flag,
                                                # post_processing_flag=False,
                                                metric_name_list=metric_name_list)

            for metric_name, metric_list in test_metric_list.items():
                record = np.array([y_list, metric_list])
                np.savetxt(osp.join(cur_performance_dir, ('raw_%s'%metric_name)+('' if layer_num is None else '_'+str(layer_num))+'.csv'), record)

            del test_dataset, test_data_loader

#======================Test test=========================================

from .parameters import params2path, get_dataset_param, get_model_param

def test_test_model(layer_num=None, batch_size=32):
    model_path = '/workspace/ldisk/IterHomoGNN/record/ShortestPathLen/random_unweighted/4_30/1000/IterGNN_10/PathConv_ihomo/Max/1/Graph/logs/Oct-11-05:14:37/model_200.pth'
    dataset_param = get_dataset_param(size=1000)
    model_param = get_model_param()
    test_dataset_param = get_dataset_param(size=100)
    test_model(model_path, dataset_param, model_param, test_dataset_param,
               layer_num=layer_num, batch_size=batch_size)

if __name__ == '__main__':
    test_test_model()
    test_test_model(layer_num=600, batch_size=8)

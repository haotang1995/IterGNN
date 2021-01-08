#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os.path as osp
import queue, os
curdir = osp.dirname(osp.abspath(__file__))
maindir = osp.dirname(curdir)

def get_all_performance_directories(prefix=None, base='performance'):
    if prefix is None:
        prefix = osp.join(maindir, 'record')
    path_list = []
    path_queue = queue.Queue()
    path_queue.put(prefix)
    while not path_queue.empty():
        path = path_queue.get()
        if osp.basename(path) == base:
            path_list.append(path)
        else:
            for fn in os.listdir(path):
                if osp.isdir(osp.join(path, fn)):
                    path_queue.put(osp.join(path, fn))
    return path_list

def max2(l):
    try:
        index = np.nanargmax(l)
        return index, l[index]
    except:
        return np.nan, 0
def min2(l):
    try:
        index = np.nanargmin(l)
        return index, l[index]
    except:
        return np.nan, np.inf
def best_parameters(test_dataset_param, layer_num=None,
                    metric_name_list=['relative_loss', 'mse_loss', 'accuracy',
                                      'post_relative_loss', 'post_mse_loss', 'post_accuracy',],
                    keys=None, no_keys=None, base='performance'):
    path_list = get_all_performance_directories(base=base)
    if keys is not None:
        path_list = [path for path in path_list if all([k in path for k in keys])]
    if no_keys is not None:
        path_list = [path for path in path_list if all([k not in path for k in no_keys])]

    # Test dataset status
    weighted_flag = (test_dataset_param.min_edge_distance != test_dataset_param.max_edge_distance)
    if weighted_flag:
        standard_weighted_flag = (abs(test_dataset_param.min_edge_distance-0.5)<1e-5) and (abs(test_dataset_param.max_edge_distance-1.5)<1e-5)
    size, index_generator = test_dataset_param.size, test_dataset_param.index_generator
    min_num_node, num_num_node = test_dataset_param.min_num_node, test_dataset_param.num_num_node

    # find all record_dir that have been tested on the given test_dataset_param
    record_list = []
    for path in path_list:
        record_dir = osp.join(path, test_dataset_param.dataset_name)
        # record_dir = osp.join(record_dir, '%s_%s'%(index_generator,'weighted' if weighted_flag else 'unweighted'))
        record_dir = osp.join(record_dir, '%s_%s'%(index_generator,(
            'weighted' if standard_weighted_flag else f'weighted_{test_dataset_param.min_edge_distance}_{test_dataset_param.max_edge_distance}'
        ) if weighted_flag else 'unweighted'))
        record_dir = osp.join(record_dir, '%d_%d'%(min_num_node, num_num_node))
        record_dir = osp.join(record_dir, '%s'%size)
        if osp.exists(record_dir):
            cur_record = []
            record_queue = queue.Queue()
            record_queue.put(record_dir)
            while not record_queue.empty():
                record_dir = record_queue.get()
                if osp.isfile(record_dir) and '.csv' in record_dir:
                    cur_record.append(record_dir)
                else:
                    for fn in os.listdir(record_dir):
                        record_queue.put(osp.join(record_dir, fn))
            record_list.append(cur_record)

    for metric_name in metric_name_list:
        cur_record_list = [[record for record in cur_record if 'raw_'+metric_name in record and '_0.csv' not in record and '_1.csv' not in record and '_2.csv' not in record and '_3.csv' not in record]
                           for cur_record in record_list]
        cur_record_list = [cur_record for cur_record in cur_record_list if len(cur_record)]

        max_cur_records = [max2([np.mean(np.loadtxt(record)[1]) for record in cur_record])
                          for cur_record in cur_record_list]
        min_cur_records = [min2([np.mean(np.loadtxt(record)[1]) for record in cur_record])
                          for cur_record in cur_record_list]

        if 'accuracy' in metric_name:
            cur_records = max_cur_records
        else:
            cur_records = min_cur_records
        index = np.argsort([cr[1] for cr in cur_records])
        print('Highest records for metric', metric_name)
        for i, ii in enumerate(index[::-1][:5]):
            if np.issubdtype(ii, np.integer) and not np.isnan(cur_records[ii][0]):
                print(i, cur_records[ii][1], cur_record_list[ii][cur_records[ii][0]],)
        index = np.argsort([cr[1] for cr in cur_records])
        print('Lowest records for metric', metric_name)
        for i, ii in enumerate(index[:5]):
            if np.issubdtype(ii, np.integer) and not np.isnan(cur_records[ii][0]):
                print(i, cur_records[ii][1], cur_record_list[ii][cur_records[ii][0]],)
        print()


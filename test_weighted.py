#!/usr/bin/env python
# coding=utf-8

import sys

from mutils import arg2dataset_param, arg2model_param, get_metric_names
from main import test_model

def main(arg, min_edge_distance=0.5, max_edge_distance=1.5):
    dataset_param = arg2dataset_param(arg[0])
    model_param = arg2model_param(arg[1])
    test_dataset_param = arg2dataset_param(arg[2])
    layer_num = None if len(arg) <= 3 else int(arg[3])

    dataset_param.size = 100000
    dataset_param.min_num_node = 4
    dataset_param.num_num_node = 30
    test_dataset_param.size = 1000
    test_dataset_param.min_num_node = 4
    test_dataset_param.num_num_node = 30
    test_dataset_param.min_edge_distance = min_edge_distance
    test_dataset_param.max_edge_distance = max_edge_distance

    metric_name_list = get_metric_names(test_dataset_param.dataset_name)
    test_model(dataset_param, model_param, test_dataset_param,
               batch_size=max(min(int(16*1000/test_dataset_param.min_num_node
                                      *10/(layer_num or model_param.layer_num)), 50), 1),
               metric_name_list=metric_name_list, layer_num=layer_num)

import copy
if __name__ == '__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1:]
    main(copy.deepcopy(arg), min_edge_distance=0.5, max_edge_distance=1.5)
    main(copy.deepcopy(arg), min_edge_distance=1, max_edge_distance=3)
    main(copy.deepcopy(arg), min_edge_distance=2, max_edge_distance=6)
    main(copy.deepcopy(arg), min_edge_distance=4, max_edge_distance=12)
    main(copy.deepcopy(arg), min_edge_distance=8, max_edge_distance=24)
    main(copy.deepcopy(arg), min_edge_distance=16, max_edge_distance=48)

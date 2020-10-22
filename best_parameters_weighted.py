#!/usr/bin/env python
# coding=utf-8

import argparse
import sys

from mutils import arg2dataset_param, arg2model_param, get_metric_names
from analysis import best_parameters

def main(args, min_edge_distance=0.5, max_edge_distance=1.5):
    test_dataset_param = arg2dataset_param(args.test_dataset_param)
    test_dataset_param.size = 1000
    test_dataset_param.min_num_node = 4
    test_dataset_param.num_num_node = 30
    test_dataset_param.min_edge_distance = min_edge_distance
    test_dataset_param.max_edge_distance = max_edge_distance
    if not args.metric:
        metric_name_list = get_metric_names(test_dataset_param.dataset_name)
    else:
        metric_name_list = args.metric
    print(test_dataset_param)

    best_parameters(test_dataset_param, metric_name_list=metric_name_list,
                    keys=args.key, no_keys=args.no_key)

import copy
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dataset_param', default='knn-weighted-100', type=str)
    parser.add_argument('--key', nargs='+', default=None, type=str)
    parser.add_argument('--no-key', nargs='+', default=None, type=str)
    parser.add_argument('--metric', nargs='+', default=None, type=str)
    args = parser.parse_args()
    for mine, maxe in [(0.5,1.5), (1,3), (2,6), (4,12), (8,24)]:
        print(f'============== Best min={mine} max={maxe} ============== ')
        main(copy.deepcopy(args), mine, maxe)

#!/usr/bin/env python
# coding=utf-8

import argparse
import sys

from mutils import arg2dataset_param, arg2model_param, get_metric_names
from analysis import best_parameters

def main(args):
    test_dataset_param = arg2dataset_param(args.test_dataset_param)
    test_dataset_param.size = 1000
    if not args.metric:
        metric_name_list = get_metric_names(test_dataset_param.dataset_name)
    else:
        metric_name_list = args.metric

    best_parameters(test_dataset_param, metric_name_list=metric_name_list,
                    keys=args.key, no_keys=args.no_key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('test_dataset_param', default='knn-weighted-100', type=str)
    parser.add_argument('--key', nargs='+', default=None, type=str)
    parser.add_argument('--no-key', nargs='+', default=None, type=str)
    parser.add_argument('--metric', nargs='+', default=None, type=str)
    args = parser.parse_args()
    main(args)

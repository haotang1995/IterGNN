#!/usr/bin/env python
# coding=utf-8

import sys

from mutils import arg2dataset_param, arg2model_param, get_metric_names
from main.test_model import test_model

def main(arg):
    dataset_param = arg2dataset_param(arg[0])
    model_param = arg2model_param(arg[1])
    test_dataset_param = arg2dataset_param(arg[2])
    layer_num = int(arg[3])
    model_path = arg[4]
    print(model_path)

    dataset_param.size = 100000
    dataset_param.min_num_node = 4
    dataset_param.num_num_node = 30
    test_dataset_param.size = 1000
    metric_name_list = get_metric_names(test_dataset_param.dataset_name)

    test_model(model_path,
               dataset_param, model_param, test_dataset_param,
               batch_size=max(min(int(16*1000/test_dataset_param.min_num_node
                                      *10/(layer_num or model_param.layer_num)), 50), 1),
               metric_name_list=metric_name_list, layer_num=layer_num)

if __name__ == '__main__':
    main(sys.argv[1:])

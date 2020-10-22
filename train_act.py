#!/usr/bin/env python
# coding=utf-8

import sys

from mutils import arg2dataset_param, arg2model_param
from main import get_general_param
from main.train_act import train_model

def main(arg):
    dataset_param = arg2dataset_param(arg[0])
    model_param = arg2model_param(arg[1])
    resume_flag = len(arg) > 2

    dataset_param.size = 100000
    dataset_param.min_num_node = 4
    dataset_param.num_num_node = 30
    general_param = get_general_param()
    general_param.save_freq = int(general_param.save_freq*10000/dataset_param.size)
    general_param.epoch_num = int(general_param.epoch_num*10000*2/dataset_param.size)
    general_param.resume_flag = resume_flag

    train_model(dataset_param, model_param, general_param)

if __name__ == '__main__':
    arg = ['knn-weighted', 'IterGNN-PathConv-Max']
    if len(sys.argv) > 1:
        arg = sys.argv[1:]
    main(arg)

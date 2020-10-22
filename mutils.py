#!/usr/bin/env python
# coding=utf-8

from main import get_dataset_param, get_model_param

SPLIT_CHAR = '-'
def arg2dataset_param(arg):
    if arg.count(SPLIT_CHAR) == 1:
        index_generator, weighted_flag = arg.split(SPLIT_CHAR)
        min_num_node = 30
    else:
        index_generator, weighted_flag, min_num_node = arg.split(SPLIT_CHAR)
        min_num_node = int(min_num_node)
    weighted_flag = (weighted_flag=='weighted')
    return get_dataset_param(index_generator=index_generator,
                             weighted_flag=weighted_flag,
                             min_num_node=min_num_node, num_num_node=1,
                             dataset_name='ShortestPathLen')
def arg2model_param(arg):
    if arg.count(SPLIT_CHAR) == 1:
        architecture_name, layer_name = arg.split(SPLIT_CHAR)
        readout_name = 'Max'
        module_num = 1
    elif arg.count(SPLIT_CHAR) == 2:
        architecture_name, layer_name, readout_name = arg.split(SPLIT_CHAR)
        module_num = 1
    else:
        architecture_name, layer_name, readout_name, module_num = arg.split(SPLIT_CHAR)
    true_architecture_name = architecture_name.strip().strip('0123456789')
    layer_num = architecture_name[len(true_architecture_name):]
    module_num = int(module_num)
    if len(layer_num):
        layer_num = int(layer_num)
    else:
        layer_num = 50
    if layer_name[:4]=='Homo':
        homogeneous_flag = 1
        layer_name = layer_name[4:]
    elif layer_name[:5]=='SHomo':
        homogeneous_flag = 2
        layer_name = layer_name[5:]
    else:
        homogeneous_flag = 0
    return get_model_param(
        architecture_name=true_architecture_name, layer_num=layer_num,
        module_num = module_num,
        layer_name=layer_name, homogeneous_flag=homogeneous_flag,
        readout_name=readout_name,
    )

def get_metric_names(dataset_name):
    metric_name_list = ['relative_loss', 'mse_loss', 'accuracy', 'prediction',
                        'post_relative_loss', 'post_mse_loss', 'post_accuracy', 'post_prediction',
                        'layer_num',]
    return metric_name_list

#=====================Testing==========================
if __name__ == '__main__':
    print(arg2model_param('IterGNN30-MPNNMaxConv-Attention'))



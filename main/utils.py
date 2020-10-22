#!/usr/bin/env python
# coding=utf-8

import torch, os, sys, logging

def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_logger(name, save_dir):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        log_file = os.path.join(save_dir, 'log.txt')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_channels(dataset):
    if dataset.__class__.__name__ == 'DatasetParam': #for debug
        in_channel, edge_channel, out_channel = 3, 1, 1
    else: #is then the dataset
        in_channel = dataset.num_node_features
        edge_channel = dataset.num_edge_features
        out_channel = dataset.num_classes
    return in_channel, edge_channel, out_channel


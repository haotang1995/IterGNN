#!/usr/bin/env python
# coding=utf-8

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.dataset import __repr__, makedirs, files_exist

import os
import os.path as osp
from shutil import rmtree

from .utils import getDevice
_curdir = osp.dirname(osp.abspath(__file__))
_fake_dataset_root = osp.join(_curdir, 'FAKEDataset')

'''
Wrapping pytorch_geometric.InMemoryDataset to support online generation of samples.
While building the datasets, instead of loading the pre-computed/collected data from disk to memory,
we generate random samples/data from a pre-defined distribution for each run of the codes.
'''
class _BaseDataset(InMemoryDataset):
    def __init__(self, size, device=None, train_flag=False,
                 transform=None, pre_filter=None, **kwargs):
        self.size, self.train_flag = size, train_flag
        self.device =device or getDevice()
        self.other_kwargs = kwargs
        self.classification_flag = False
        super(InMemoryDataset, self).__init__(_fake_dataset_root, transform=transform,
                                           pre_transform=None, pre_filter=pre_filter)
        # self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return ['raw_files.pt']
    @property
    def processed_file_names(self):
        return ['data.pt']
    def download(self):
        pass
    def generate_data_list(self):
        '''
        generate data list here
        '''
        raise NotImplementedError
    def process(self):
        data_list = self.generate_data_list()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        self.data, self.slices = self.collate(data_list)
        self.data, self.slices = self.data.to(self.device), {k:v.to(self.device) for k,v in self.slices.items()}
        # torch.save((data, slices), self.processed_paths[0])
    def _process(self):
        if files_exist(self.processed_paths):
            for _path in self.processed_paths:
                os.remove(_path)
        super(_BaseDataset, self)._process()
    def _create(self, idx):
        '''
        create one sample
        '''
        raise NotImplementedError

class GraphDataset(_BaseDataset):
    def generate_data_list(self,):
        data_list = [self._create(idx) for idx in range(self.size)]
        return data_list

#================== Testing =================
from torch_geometric.data import Data
import numpy as np

def _test_indexing_slicing(dataset):
    size = len(dataset)
    # test indexing
    for index in np.random.randint(low=0, high=size, size=[10]).tolist():
        assert(dataset[index].x == dataset[index].x)
    # test slicing
    for index in np.random.randint(low=0, high=size, size=[10,2]).tolist():
        index = slice(*(sorted(index)))
        assert(torch.prod(dataset[index].data.x == dataset[index].data.x))
def _test_dataset(dataset_class):
    # test generating
    dataset = dataset_class(1)
    assert(len(dataset) == 1)
    dataset = dataset_class(10)
    assert(len(dataset) == 10)
    dataset = dataset_class(100)
    assert(len(dataset) == 100)
    size = np.random.randint(10000)
    dataset = dataset_class(size)
    assert(len(dataset) == size)

    _test_indexing_slicing(dataset)

    # test shuffle
    dataset.shuffle()

    _test_indexing_slicing(dataset)

class _TestedGraphDataset(GraphDataset):
    def _create(self, idx):
        return Data(
            x = torch.rand([1,1], device=self.device),
            edge_index = torch.zeros([2,0], device=self.device, dtype=torch.long)
        )
def test_graph_dataset():
    _test_dataset(_TestedGraphDataset)

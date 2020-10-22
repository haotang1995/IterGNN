#!/usr/bin/env python
# coding=utf-8

import numpy as np
import torch, queue
from torch_geometric.data import Data

from .utils import gen_edge_index
from .base import GraphDataset

def _shortest_path_len(edge_index, edge_distance, src, tar):
    device = edge_index.device
    rol, col = edge_index

    q = queue.PriorityQueue()
    visited_node_set = set()

    # standard Dijkstra's algorithm
    done = False
    q.put((0, torch.tensor(src, device=device)))
    while (not done and not q.empty()):
        length, current_node = q.get()
        if current_node.item() not in visited_node_set:
            visited_node_set.add(current_node.item())
            if tar in current_node:
                done = True
            else:
                mask = rol == current_node
                for node, weight in zip(col[mask].unbind(), edge_distance[mask].unbind()):
                    if node.item() not in visited_node_set:
                        q.put((length+weight, node))
    length = torch.tensor(length.item() if done else -1, dtype=torch.float, device=device).reshape([1])

    return length

def _generate_sample(node_num=100, sparsity=0.5, k=8, dim=2, lobster_prob=(0.2,0.2),
                     connect_prob=0.01, index_generator='random', device=None,
                     min_edge_distance=1., max_edge_distance=1., **kwargs):
    edge_index, node_num = gen_edge_index(index_generator, node_num=node_num,
                                          sparsity=sparsity, k=k, dim=dim,
                                          lobster_prob=lobster_prob,
                                          connect_prob=connect_prob,
                                          device=device)
    device = edge_index.device

    edge_distance = torch.rand([edge_index.size(1),1],device=device)*(max_edge_distance-min_edge_distance)+min_edge_distance
    src, tar = np.random.choice(node_num, [2,], replace=False)
    node_feat = torch.tensor([[0,0,1] if i != src and i != tar else ([1,0,0] if i == src else [0,1,0]) for i in range(node_num)], dtype=torch.float, device=device)
    length = _shortest_path_len(edge_index, edge_distance, src, tar)
    if max_edge_distance-min_edge_distance > 0.5:
        node_feat = node_feat * (max_edge_distance-min_edge_distance)
    return Data(x=node_feat, edge_index=edge_index, edge_attr=edge_distance, y=length)

class ShortestPathLenDataset(GraphDataset):
    def _create(self, idx):
        if idx < 0 or idx >= self.size:
            raise IndexError
        min_num_node, num_num_node = self.other_kwargs['min_num_node'], self.other_kwargs['num_num_node']
        node_num = int(idx/self.size*num_num_node+min_num_node)
        data = _generate_sample(node_num=node_num, **self.other_kwargs)
        while (data.y < 0):
            data = _generate_sample(node_num=node_num, **self.other_kwargs)
        return data
    @property
    def num_classes(self):
        return 1

#========= Testing ============

import time
from .utils import print_list_properties
import matplotlib.pyplot as plt
plt.ion()

def _test_distance_dataset(index_generator, min_edge_distance, max_edge_distance):
    size = 1000

    # generating dataset
    start_time = time.time()
    dataset_generator = lambda **kwargs: ShortestPathLenDataset(size=size, index_generator=index_generator,
                                                              min_edge_distance=min_edge_distance,
                                                              max_edge_distance=max_edge_distance,
                                                              min_num_node=4, num_num_node=100,
                                                              **kwargs)
    if index_generator == 'random':
        dataset = dataset_generator(sparsity=0.5)
    elif index_generator == 'knn':
        dataset = dataset_generator(k=8, dim=1)
    elif index_generator == 'mesh':
        dataset = dataset_generator(dim=2)
    elif index_generator == 'lobster':
        dataset = dataset_generator(lobster_prob=(0.2, 0.2))
    else:
        raise NotImplementedError('No index generator named %s can be tested'%index_generator)
    print('** It takes %.2f seconds to generate the dataset'%(time.time()-start_time))

    # analyse dataset's properties --> distance
    distance_list = [d.y.item() for d in dataset]
    print_list_properties(distance_list, 'distance')
    fig = plt.figure()
    plt.hist(distance_list, bins=50)
    plt.title('%s_%s'%(index_generator, 'weighted' if min_edge_distance != max_edge_distance else 'unweighted'))
    plt.show()

def test_random_unweighted():
    _test_distance_dataset('random', 1., 1.)
def test_random_weighted():
    _test_distance_dataset('random', 0.5, 1.5)
def test_knn_unweighted():
    _test_distance_dataset('knn', 1., 1.)
def test_knn_weighted():
    _test_distance_dataset('knn', 0.5, 1.5)
def test_mesh_unweighted():
    _test_distance_dataset('mesh', 1., 1.)
def test_mesh_weighted():
    _test_distance_dataset('mesh', 0.5, 1.5)
def test_lobster_unweighted():
    _test_distance_dataset('lobster', 1., 1.)
def test_lobster_weighted():
    _test_distance_dataset('lobster', 0.5, 1.5)
def main():
    objects = {k:v for k,v in globals().items() if k[:5] == 'test_'}
    for k,v in objects.items():
        print()
        print('====Running %s====='%k)
        v()
    plt.ioff()
    plt.show()

if __name__ == '__main__':
    main()

import torch, time, itertools
import numpy as np
import networkx as nx

from scipy.spatial import Delaunay

from torch_sparse import coalesce
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.nn import knn_graph, radius_graph

def block_index(n, d, i):
    return int(n*i*1./d)
def block_size(n, d, i):
    return block_index(n,d,i+1)-block_index(n,d,i)

def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def build_fully_connected_graph(N, device=None):
    index = torch.arange(N, device=device).unsqueeze(-1).expand([-1,N])
    rol = torch.reshape(index, [-1])
    col = torch.reshape(torch.t(index), [-1])
    edge_index = torch.stack([rol, col], dim=0)
    return edge_index

'''
edge index generator
'''
def gen_random_index(node_num=100, sparsity=0.5, device=None, **kwargs):
    device = device or getDevice()
    edge_index = erdos_renyi_graph(node_num, 1-sparsity, directed=True)
    return edge_index.to(device).type(torch.long), node_num
def gen_undirected_random_index(node_num=100, sparsity=0.5, device=None, **kwargs):
    device = device or getDevice()
    edge_index = erdos_renyi_graph(node_num, 1-sparsity, directed=False)
    return edge_index.to(device).type(torch.long), node_num
def gen_knn_index(node_num=100, dim=1, k=8, device=None, **kwargs):
    device = device or getDevice()
    pos = torch.rand([node_num, dim], dtype=torch.float, device=device)
    edge_index = knn_graph(pos, k, )
    return edge_index, node_num
def gen_mesh_index(node_num=100, dim=2, device=None, **kwargs):
    n, d = node_num, dim
    n = max(n, d+3)
    device = device or getDevice()

    points = np.random.rand(n, d)
    tri = Delaunay(points, qhull_options='QJ')
    edge_index = [e for tett in tri.simplices for e in itertools.combinations(tett, 2)]
    edge_index = list(set(edge_index))
    edge_index = list(zip(*edge_index))
    rol, col = edge_index
    edge_index = [rol+col, col+rol]
    edge_index = torch.tensor(edge_index, dtype=torch.long, device=torch.device('cpu'))
    edge_index, _ = coalesce(edge_index, torch.zeros_like(edge_index)[0], n, n )
    return edge_index.to(device), n
def gen_lobster_index(node_num=100, lobster_prob=(0.2, 0.2), device=None, **kwargs):
    device = device or getDevice()
    p1, p2 = lobster_prob
    n = node_num

    path_edges = [list(range(n-1)), list(range(1,n))]
    first_leaf_num = np.random.binomial(n, p1)
    first_leaf_edges = [np.random.randint(low=0, high=n, size=first_leaf_num).tolist(),
                        np.arange(n, n+first_leaf_num).tolist()]
    second_leaf_num = np.random.binomial(first_leaf_num, p2)
    second_leaf_edges = [np.random.randint(low=n, high=n+first_leaf_num, size=second_leaf_num).tolist(),
                         np.arange(n+first_leaf_num, n+first_leaf_num+second_leaf_num).tolist()]

    rol = path_edges[0]+first_leaf_edges[0]+second_leaf_edges[0]
    col = path_edges[1]+first_leaf_edges[1]+second_leaf_edges[1]

    edges = [rol+col, col+rol]
    edge_index = torch.tensor(edges, dtype=torch.long, device=device)
    return edge_index, n+first_leaf_num+second_leaf_num
def gen_tree_index(node_num, device=None):
    device = device or getDevice()
    graph = nx.from_prufer_sequence(np.random.randint(low=0, high=node_num, size=[node_num-2]))
    edge_index = torch.tensor(list(zip(*(graph.edges))), dtype=torch.long, device=device)
    edge_index = torch.cat([edge_index, edge_index[:,[1,0]]], dim=-1)
    # edge_index, _ = coalesce(edge_index, None, node_num, node_num )
    edge_index, _ = coalesce(edge_index, torch.zeros_like(edge_index)[0], node_num, node_num )
    return edge_index, node_num
def gen_line_index(node_num, device=None, *args, **kwargs):
    device = device or getDevice()
    rol = torch.arange(node_num-1, device=device)
    col = torch.arange(1, node_num, device=device)
    edge_index = torch.cat([
        torch.stack([rol, col], dim=0),
        torch.stack([col, rol], dim=0),
    ], dim=-1)
    return edge_index, node_num
def gen_grid_index(node_num, device=None, *args, **kwargs):
    index = torch.arange(node_num*node_num, dtype=torch.long, device=device)
    grid_index = index.reshape([node_num, node_num])
    down_index = torch.stack([
        grid_index[:-1].reshape(-1),
        grid_index[1:].reshape(-1),
    ], dim=0)
    right_index = torch.stack([
        grid_index[:,:-1].reshape(-1),
        grid_index[:,1:].reshape(-1),
    ], dim=0)
    edge_index = torch.cat([down_index, right_index], dim=-1)
    edge_index = torch.cat([edge_index, edge_index[[1,0]]],dim=-1)
    return edge_index, node_num*node_num
def gen_edge_index(index_generator='random', **kwargs):
    return globals()['gen_%s_index'%index_generator](**kwargs)

#=============== Testing ===============

import multiprocessing as mp
from torch_geometric.data import Data
from torch_geometric.utils import scatter_, to_networkx
import networkx as nx
import matplotlib.pyplot as plt
plt.ion()

POOL_NUM = 1
pool = mp.Pool(POOL_NUM)

def print_list_properties(l, name):
    print('%s:\tmax %.2f, min %.2f, mean %.2f, std %.2f, median %.2f'
          %(name, np.max(l), np.min(l), np.mean(l), np.std(l), np.median(l)))
def _g2nx(edge_index):
    assert(type(edge_index) == list)
    g, ug = nx.DiGraph(), nx.Graph()
    g.add_edges_from(edge_index)
    print(g.number_of_nodes())
    # ug.add_edges(edge_index)
    return g
def _shortest_path_length(g):
    src, tgt = np.random.randint(g.number_of_nodes(), size=2)
    for _ in range(100):
        try:
            return nx.shortest_path_length(g, source=src, target=tgt)
        except:
            pass
    return 0
def _test_edge_index_generator(index_generator, device):
    num = 1000
    min_node_num, max_node_num = 4, 100

    # generation
    start_time = time.time()
    graphs = [index_generator(node_num=node_num) for node_num in
              np.random.randint(min_node_num, max_node_num, num).tolist()]
    print('** It takes %.2f seconds to generate %d graphs'%(time.time()-start_time, num))

    # node number distribution
    node_num_list = [node_num for _, node_num in graphs]
    print_list_properties(node_num_list, 'node_num')

    # edge number distribution
    edge_num_list = [edge_index.size(1) for edge_index,_ in graphs]
    print_list_properties(edge_num_list, 'edge_num')

    # sparsity distribution
    sparsity_list = [edge_index.size(1)*1./node_num/(node_num-1)
                     for edge_index, node_num in graphs]
    print_list_properties(sparsity_list, 'sparsity')

    # degree distribution
    outer_degree_list = [deg for edge_index, node_num in graphs
                         for deg in scatter_('add', torch.ones_like(edge_index[0]),
                                             edge_index[0], dim=0, dim_size=node_num).tolist()]
    print_list_properties(outer_degree_list, 'outer_degree')
    inner_degree_list = [deg for edge_index, node_num in graphs
                         for deg in scatter_('add', torch.ones_like(edge_index[1]),
                                             edge_index[1], dim=0, dim_size=node_num).tolist()]
    print_list_properties(inner_degree_list, 'inner_degree')

    # convert to networkx graphs
    start_time = time.time()
    nx_graphs = [to_networkx(Data(x=torch.zeros([node_num,1],device=edge_index.device),
                                  edge_index=edge_index))
                 for edge_index, node_num in graphs]
    print('** It takes %.2f seconds to convert to networkx graphs'%(time.time()-start_time))

    # component distribution
    component_number_list = [nx.number_weakly_connected_components(g) for g in nx_graphs]
    print_list_properties(component_number_list, 'component_num')

    # planarity distribution
    planarity_list = [nx.check_planarity(g)[0]*1. for g in nx_graphs]
    print_list_properties(planarity_list, 'planarity')

    # path distribution
    path_length_list = [_shortest_path_length(g) for g in nx_graphs for _ in range(30)]
    print_list_properties(path_length_list, 'path_length')
    fig = plt.figure()
    plt.hist(path_length_list, bins=50)
    plt.title(str(index_generator.__name__))
    plt.show()

def test_random_cpu():
    _test_edge_index_generator(gen_random_index, torch.device('cpu'))
def test_random_gpu():
    _test_edge_index_generator(gen_random_index, torch.device('cuda'))
def test_undirected_random_cpu():
    _test_edge_index_generator(gen_undirected_random_index, torch.device('cpu'))
def test_undirected_random_gpu():
    _test_edge_index_generator(gen_undirected_random_index, torch.device('cuda'))
def test_knn_cpu():
    _test_edge_index_generator(gen_knn_index, torch.device('cpu'))
def test_knn_gpu():
    _test_edge_index_generator(gen_knn_index, torch.device('cuda'))
def test_mesh_cpu():
    _test_edge_index_generator(gen_mesh_index, torch.device('cpu'))
def test_mesh_gpu():
    _test_edge_index_generator(gen_mesh_index, torch.device('cuda'))
def test_lobster_cpu():
    _test_edge_index_generator(gen_lobster_index, torch.device('cpu'))
def test_lobster_gpu():
    _test_edge_index_generator(gen_lobster_index, torch.device('cuda'))
def test_tree_cpu():
    _test_edge_index_generator(gen_tree_index, torch.device('cpu'))
def test_tree_gpu():
    _test_edge_index_generator(gen_tree_index, torch.device('cuda'))

if __name__ == '__main__':
    objects = {k:v for k,v in globals().items() if k[:5] == 'test_'}
    for k,v in objects.items():
        print()
        print('====Running %s====='%k)
        v()
    plt.ioff()
    plt.show()

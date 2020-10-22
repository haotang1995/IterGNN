#!/usr/bin/env python
# coding=utf-8

import torch, copy
from tqdm import tqdm
from torch_geometric.data import Data, Batch

def torch2numpy(tensor):
    return tensor.detach().cpu().numpy()
def torch2list(tensor):
    return tensor.detach().cpu().reshape([-1]).tolist()
    # return torch2numpy(tensor).reshape([-1]).tolist()

def relative_loss(y, pred, layer_num):
    return torch2list(torch.abs((pred.reshape(y.size())-y)/y))
def mse_loss(y, pred, layer_num):
    return torch2list((pred.reshape(y.size())-y)**2)
def accuracy(y, pred, layer_num):
    result = torch2list((torch.abs(y-pred.reshape(y.size()))<1e-4).type(torch.float))
    return result
def intaccuracy(y, pred, layer_num):
    y, pred = torch.round(y), torch.round(pred)
    return accuracy(y, pred, layer_num)
def layer_num(y, pred, layer_num):
    return [float(layer_num)]*y.reshape(-1).size(0)
def prediction(y, pred, layer_num):
    return torch2list(pred)

# The function should perform the same as pytorch_geometric.data.Batch.to_data_list
# Implement here for Batch that is not created by from_data_list
# (Note that this function assumes no permutation after creating the Batch. So,
# the node indexes for each graph are continuous.)
def to_data_list(data):
    if 'to_data_list' in data.__dict__:
        return data.to_data_list()
    graph_indexes = set(data.batch.tolist())
    data_list = []
    for gi in graph_indexes:
        node_indexes = torch.arange(data.x.size(0), device=data.x.device)
        node_indexes = node_indexes[data.batch == gi]
        node_index_max, node_index_min = torch.max(node_indexes), torch.min(node_indexes)
        edge_indexes = (data.edge_index[0]>=node_index_min)&(data.edge_index[0]<=node_index_max)
        edge_indexes = torch.arange(data.edge_index.size(1), device=data.x.device)[edge_indexes]

        x = data.x[node_indexes]
        edge_index = data.edge_index[:,edge_indexes]-node_index_min
        edge_attr = data.edge_attr[edge_indexes]
        y = data.y[gi:gi+1]
        batch = torch.zeros_like(node_indexes)

        data_list.append(Batch(
            x=x, y=y, batch=batch,
            edge_index=edge_index, edge_attr=edge_attr,
        ))
    assert(data.x.size(0)==sum([d.x.size(0) for d in data_list]))
    assert(all([d.x.size(0) == d.batch.size(0) for d in data_list]))
    assert(data.edge_index.size(1)==sum([d.edge_index.size(1) for d in data_list]))
    assert(all([d.edge_index.size(1)==d.edge_attr.size(0) for d in data_list]))
    assert(all([data.x.size(1) == d.x.size(1) for d in data_list]))
    assert(all([data.edge_attr.size(1) == d.edge_attr.size(1) for d in data_list]))
    return data_list
# The Post-processing function to find an exact path from src to tar using the
# learned shortest-path predictor.
# More details are available at Section D.1.2. in
# https://haotang1995.github.io/files/IterGNN_appendix.pdf
def post_processing(net, data):
    data = copy.deepcopy(data)
    device = data.x.device

    node_num = data.x.size(0)
    rol, col = data.edge_index
    src, tar = torch.argmax(data.x[:,0]).item(), torch.argmax(data.x[:,1]).item()
    edge_weights = data.edge_attr.reshape([-1])

    visited_nodes = set()
    current_path_length, current_target = 0., tar
    pred, = net(data)
    visited_nodes.add(current_target)
    while current_target != src and pred.item() > 1e-7:
        candidates = rol[col==current_target]
        weights = edge_weights[col==current_target]
        if torch.sum(candidates==src) > 0:
            current_path_length += weights[candidates==src].item()
            break;
        cand_pred = torch.zeros_like(candidates, dtype=torch.float).reshape([-1])
        for ci, cand in enumerate(candidates):
            data.x = torch.tensor([[0,0,1] if i != src and i != cand
                                    else ([1,0,0] if i == src else [0,1,0]) for i in range(node_num)],
                                    dtype=torch.float, device=device)
            cand_pred[ci], = net(data)
        mask = cand_pred < pred.item()
        if torch.sum(mask) == 0:
            print('PostProcessing Error: no candidate is on the path')
            mask = torch.tensor([cd not in visited_nodes for cd in candidates.tolist()],
                                dtype=mask.dtype, device=mask.device)
            if torch.sum(mask) == 0:
                break;
        candidates, cand_pred, weights = candidates[mask], cand_pred[mask], weights[mask].reshape([-1])
        ci = torch.argmin(torch.abs(cand_pred+weights-pred.item())).item()
        pred = cand_pred[ci]
        current_path_length += weights[ci].item()
        current_target = candidates[ci].item()
        visited_nodes.add(current_target)
    return current_path_length

def evaluate(net, loader, device, parallel_flag=False,
             metric_name_list=['relative_loss']):
    net.eval()
    y_list = []
    metric_list = {mn:[] for mn in metric_name_list}
    post_processing_flag = any([mn[:5]=='post_' for mn in metric_name_list])
    for data in tqdm(loader):
        if not parallel_flag:
            data = data.to(device)
        if parallel_flag:
            y = torch.cat([d.y for d in data]).to(device)
        else:
            y = data.y

        pred, layer_num = net(data, output_layer_num_flag=True)
        if post_processing_flag:
            data_list = to_data_list(data)
            pred_list = [post_processing(net, d) for d in data_list]
            post_pred = torch.tensor(pred_list, device=device, dtype=torch.float)
        for mn, ml in metric_list.items():
            post_flag = mn[:5]=='post_'
            cpred = post_pred if post_flag else pred
            cmetric = mn[5:] if post_flag else mn
            ml += globals()[cmetric](y, cpred, layer_num)
        y_list += torch2list(data.y)
    return metric_list, y_list


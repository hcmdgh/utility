import torch 
from torch import Tensor 
from torch_geometric.data import Data, Batch 
from contextlib import suppress
from typing import Any 


def get_pyg_data_num_nodes(data: Any) -> int:
    if 'num_nodes' in data:
        return int(data['num_nodes'])
    elif 'x' in data:
        return len(data['x']) 
    else:
        raise ValueError


def get_pyg_data_node_feat(data: Any) -> Tensor:
    num_nodes = get_pyg_data_num_nodes(data)
    node_feat_2d = data['x']
    node_feat_dim = node_feat_2d.shape[-1]
    assert node_feat_2d.dtype == torch.float32 
    assert node_feat_2d.shape == (num_nodes, node_feat_dim)

    return node_feat_2d 


def get_pyg_data_edge_index(data: Any) -> Tensor:
    edge_index_2d = data['edge_index']
    num_edges = edge_index_2d.shape[1]
    assert edge_index_2d.dtype == torch.int64 
    assert edge_index_2d.shape == (2, num_edges)

    return edge_index_2d 


def get_pyg_data_node_feat_dim(data: Any) -> int:
    return get_pyg_data_node_feat(data).shape[-1]


def get_pyg_data_node_label(data: Any) -> Tensor:
    num_nodes = get_pyg_data_num_nodes(data)
    node_label = data['y']
    assert node_label.dtype in [torch.int64, torch.float32, torch.bool]
    assert len(node_label) == num_nodes 

    return node_label 


def get_pyg_data_node_num_classes(data: Any) -> int:
    node_label = get_pyg_data_node_label(data)

    if node_label.dtype == torch.int64:
        return int(node_label.max()) + 1
    elif node_label.dtype == torch.float32:
        return 1 
    else:
        raise TypeError 
    

def get_pyg_data_graph_ptr(data: Any) -> Tensor:
    ptr = data['ptr']
    assert ptr.dtype == torch.int64
    assert ptr.ndim == 1 

    return ptr 


def get_pyg_data_num_graphs(data: Any) -> int:
    ptr = get_pyg_data_graph_ptr(data)
    num_graphs = len(ptr) - 1 

    return num_graphs 


def get_pyg_data_graph_label(data: Any) -> Tensor:
    num_graphs = get_pyg_data_num_graphs(data)
    graph_label = data['y']
    assert graph_label.dtype in [torch.int64, torch.float32, torch.bool]
    assert len(graph_label) == num_graphs 

    return graph_label 


def get_pyg_data_graph_num_classes(data: Any) -> int:
    graph_label = get_pyg_data_graph_label(data)

    if graph_label.dtype == torch.int64:
        return int(graph_label.max()) + 1
    elif graph_label.dtype == torch.float32:
        return 1 
    else:
        raise TypeError 
    

def get_pyg_data_node_or_graph_label(data: Any) -> Tensor:
    with suppress(Exception):
        return get_pyg_data_node_label(data)
    
    with suppress(Exception):
        return get_pyg_data_graph_label(data)

    raise ValueError 


def get_pyg_data_node_or_graph_num_classes(data: Any) -> int:
    with suppress(Exception):
        return get_pyg_data_node_num_classes(data)
    
    with suppress(Exception):
        return get_pyg_data_graph_num_classes(data)

    raise ValueError 


def get_pyg_data_node_train_mask(data: Any) -> Tensor:
    num_nodes = get_pyg_data_num_nodes(data)
    node_train_mask = data['train_mask']
    assert node_train_mask.dtype == torch.bool
    assert node_train_mask.shape == (num_nodes,) 

    return node_train_mask


def get_pyg_data_node_val_mask(data: Any) -> Tensor:
    num_nodes = get_pyg_data_num_nodes(data)
    node_val_mask = data['val_mask']
    assert node_val_mask.dtype == torch.bool
    assert node_val_mask.shape == (num_nodes,) 

    return node_val_mask


def get_pyg_data_node_test_mask(data: Any) -> Tensor:
    num_nodes = get_pyg_data_num_nodes(data)
    node_test_mask = data['test_mask']
    assert node_test_mask.dtype == torch.bool
    assert node_test_mask.shape == (num_nodes,) 

    return node_test_mask


def get_pyg_data_dense_adj_mat(data: Any) -> Tensor: 
    num_nodes = int(data['num_nodes'])
    edge_index = data['edge_index']
    num_edges = edge_index.shape[1]
    assert edge_index.shape == (2, num_edges)
    device = edge_index.device

    adj_mat = torch.sparse_coo_tensor(
        indices = edge_index,
        values = torch.ones(num_edges, dtype=torch.float32, device=device),
        size = (num_nodes, num_nodes),
    )

    adj_mat_dense = adj_mat.to_dense()
    assert adj_mat_dense.layout == torch.strided
    assert adj_mat_dense.shape == (num_nodes, num_nodes)

    return adj_mat_dense 

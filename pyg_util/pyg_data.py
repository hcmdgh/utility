import torch 
from torch import Tensor 
import torch_geometric.utils as pyg_util 
from typing import Any 


class PyGDataWrapper:
    def __init__(
        self,
        data: Any,
    ):
        self.data = data 

    def __str__(self) -> str:
        return self.data.__str__()
    
    def __repr__(self) -> str:
        return self.data.__repr__()
    
    @property 
    def num_nodes(self) -> int:
        return int(self.data.num_nodes)
        
    @property
    def num_edges(self) -> int:
        return int(self.data.num_edges)
    
    @property 
    def device(self) -> torch.device:
        return self.data['edge_index'].device 
    
    def to(
        self,
        device: Any,
    ) -> 'PyGDataWrapper':
        device = torch.device(device)
        self.data = self.data.to(device)

        return self 

    @property
    def node_feat(self) -> Tensor:
        return self.data['x']
    
    @property
    def edge_index(self) -> Tensor:
        return self.data['edge_index']
    
    @property
    def node_feat_dim(self) -> int:
        assert self.node_feat.ndim == 2 

        return self.node_feat.shape[-1]
    
    @property 
    def node_or_graph_label(self) -> Tensor:
        return self.data['y']
    
    node_label = node_or_graph_label

    @property
    def node_or_graph_num_classes(self) -> int:
        node_or_graph_label = self.node_or_graph_label
        assert node_or_graph_label.dtype == torch.int64 and node_or_graph_label.ndim == 1 

        return int(node_or_graph_label.max()) + 1

    node_num_classes = node_or_graph_num_classes
    
    @property 
    def graph_ptr(self) -> Tensor:
        return self.data['ptr']
    
    @property
    def num_graphs(self) -> int:
        ptr = self.graph_ptr 
        assert ptr.ndim == 1 

        return len(ptr) - 1

    @property 
    def node_train_mask(self) -> Tensor:
        return self.data['train_mask']
    
    @property 
    def node_val_mask(self) -> Tensor:
        return self.data['val_mask']
    
    @property 
    def node_test_mask(self) -> Tensor:
        return self.data['test_mask']
    
    def dense_adj_mat(self) -> Tensor:
        adj_mat = torch.sparse_coo_tensor(
            indices = self.edge_index,
            values = torch.ones(self.num_edges, dtype=torch.float32, device=self.device),
            size = (self.num_nodes, self.num_nodes),
        )

        adj_mat_dense = adj_mat.to_dense()
        assert adj_mat_dense.layout == torch.strided
        assert adj_mat_dense.shape == (self.num_nodes, self.num_nodes)

        return adj_mat_dense 

    def add_self_loop_(self) -> 'PyGDataWrapper':
        self.data.edge_index, _ = pyg_util.add_remaining_self_loops(
            edge_index = self.data.edge_index,
            num_nodes = int(self.data.num_nodes),
        )

        return self 

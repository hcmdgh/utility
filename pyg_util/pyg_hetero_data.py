import torch 
from torch import Tensor 
from typing import Any


class PyGHeteroDataWrapper:
    def __init__(
        self,
        hetero_data: Any,
    ):
        self.hetero_data = hetero_data 

    def __str__(self) -> str:
        return self.hetero_data.__str__()
    
    def __repr__(self) -> str:
        return self.hetero_data.__repr__()
    
    @property 
    def num_nodes(self) -> int:
        return int(self.hetero_data.num_nodes)
        
    @property
    def num_edges(self) -> int:
        return int(self.hetero_data.num_edges)
    
    @property 
    def device(self) -> torch.device:
        return next(iter(self.hetero_data.edge_index_dict.values())).device 
    
    def to(
        self,
        device: Any,
    ) -> 'PyGHeteroDataWrapper':
        device = torch.device(device)
        self.hetero_data = self.hetero_data.to(device)

        return self 

    @property
    def node_type_set(self) -> set[str]:
        return set(self.hetero_data.node_types)
    
    @property
    def edge_type_set(self) -> set[tuple[str, str, str]]:
        return set(self.hetero_data.edge_types)

    @property
    def num_nodes_dict(self) -> dict[str, int]:
        return self.hetero_data.num_nodes_dict
    
    @property
    def node_feat_dict(self) -> dict[str, Tensor]:
        return self.hetero_data.x_dict
    
    @property
    def node_feat_dim_dict(self) -> dict[str, int]:
        return {
            node_type: node_feat.shape[-1]
            for node_type, node_feat in self.hetero_data.x_dict.items() 
        }
    
    @property
    def edge_index_dict(self) -> dict[tuple[str, str, str], Tensor]:
        return self.hetero_data.edge_index_dict

    @property
    def target_node_type(self) -> str:
        target_node_type_list = list(self.hetero_data.y_dict.keys()) 
        assert len(target_node_type_list) == 1 

        return target_node_type_list[0]
    
    @property 
    def target_num_nodes(self) -> int:
        return self.num_nodes_dict[self.target_node_type]

    @property
    def target_node_feat(self) -> Tensor:
        return self.node_feat_dict[self.target_node_type]
    
    @property
    def target_node_feat_dim(self) -> int:
        target_node_feat = self.target_node_feat 
        assert target_node_feat.ndim == 2

        return target_node_feat.shape[-1]

    @property
    def target_node_label(self) -> Tensor:
        node_label_list = list(self.hetero_data.y_dict.values())
        assert len(node_label_list) == 1 

        return node_label_list[0] 

    @property
    def target_node_num_classes(self) -> int:
        node_label = self.target_node_label
        assert node_label.ndim == 1 and node_label.dtype == torch.int64 

        return int(node_label.max()) + 1 

    @property 
    def target_node_train_mask(self) -> Tensor:
        return self.hetero_data[self.target_node_type].train_mask
    
    @property 
    def target_node_val_mask(self) -> Tensor:
        return self.hetero_data[self.target_node_type].val_mask
    
    @property 
    def target_node_test_mask(self) -> Tensor:
        return self.hetero_data[self.target_node_type].test_mask

    @property
    def short_edge_type_2_edge_type(self) -> dict[str, tuple[str, str, str]]:
        edge_type_set = self.edge_type_set

        short_etype_2_etype = {
            edge_type[1]: edge_type
            for edge_type in edge_type_set
        }
        assert len(short_etype_2_etype) == len(edge_type_set) 

        return short_etype_2_etype 

    def short_metapath_list_2_complete_metapath_list(
        self, 
        short_metapath_list: list[list[str]],
    ) -> list[list[tuple[str, str, str]]]:
        short_edge_type_2_edge_type = self.short_edge_type_2_edge_type

        complete_metapath_list = [
            [
                short_edge_type_2_edge_type[short_edge_type] 
                for short_edge_type in metapath 
            ]
            for metapath in short_metapath_list
        ]

        return complete_metapath_list 

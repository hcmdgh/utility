import torch 
from torch import Tensor 
from torch_geometric.data import HeteroData 
from torch_sparse import SparseTensor


def get_hetero_graph_node_type_list(
    graph: HeteroData,
) -> list[str]:
    node_type_list = graph.node_types
    assert isinstance(node_type_list, list)

    return node_type_list 


def get_hetero_graph_node_type_set(
    graph: HeteroData,
) -> set[str]:
    return set(get_hetero_graph_node_type_list(graph=graph))


def get_hetero_graph_edge_type_list(
    graph: HeteroData,
) -> list[tuple[str, str, str]]:
    edge_type_list = graph.edge_types
    assert isinstance(edge_type_list, list)

    return edge_type_list 


def get_hetero_graph_edge_type_set(
    graph: HeteroData,
) -> set[tuple[str, str, str]]:
    return set(get_hetero_graph_edge_type_list(graph=graph))


def get_hetero_graph_num_nodes_dict(
    graph: HeteroData,
) -> dict[str, int]:
    num_nodes_dict = graph.num_nodes_dict
    assert isinstance(num_nodes_dict, dict)

    return num_nodes_dict 


def get_hetero_graph_target_node_type(
    graph: HeteroData,
) -> str:
    target_node_types = list(graph.y_dict.keys()) 
    assert len(target_node_types) == 1 

    return target_node_types[0]


def get_hetero_graph_node_feat_dim_dict(
    graph: HeteroData,
) -> dict[str, int]:
    return {
        node_type: node_feat.shape[-1]
        for node_type, node_feat in graph.x_dict.items() 
    }


def get_hetero_graph_node_feat_dict(
    graph: HeteroData,
) -> dict[str, Tensor]:
    node_feat_dict = graph.x_dict 
    assert isinstance(node_feat_dict, dict)

    return node_feat_dict


def get_hetero_graph_edge_index_dict(
    graph: HeteroData,
) -> dict[tuple[str, str, str], Tensor]:
    edge_index_dict = graph.edge_index_dict 
    assert isinstance(edge_index_dict, dict)

    return edge_index_dict


def get_hetero_graph_target_node_feat(
    graph: HeteroData,
) -> Tensor:
    target_node_type = get_hetero_graph_target_node_type(graph)
    node_feat = graph[target_node_type].x 
    assert isinstance(node_feat, Tensor) 

    return node_feat 


def get_hetero_graph_target_node_feat_dim(
    graph: HeteroData,
) -> int:
    node_feat = get_hetero_graph_target_node_feat(graph=graph)

    return node_feat.shape[-1]


def get_hetero_graph_target_node_label(
    graph: HeteroData,
) -> Tensor:
    node_label_list = list(graph.y_dict.values())
    assert len(node_label_list) == 1 

    return node_label_list[0] 


def get_hetero_graph_target_num_nodes(
    graph: HeteroData,
) -> int:
    node_label = get_hetero_graph_target_node_label(graph)

    return len(node_label)


def get_hetero_graph_target_node_num_classes(
    graph: HeteroData,
) -> int:
    node_label_1d = get_hetero_graph_target_node_label(graph)
    assert node_label_1d.ndim == 1 and node_label_1d.dtype == torch.int64 

    num_classes = int(node_label_1d.max()) + 1 

    return num_classes 


def get_hetero_graph_target_node_train_mask(
    graph: HeteroData,
) -> Tensor:
    node_train_mask_list = list(graph.train_mask_dict.values())
    assert len(node_train_mask_list) == 1 

    return node_train_mask_list[0] 


def get_hetero_graph_target_node_val_mask(
    graph: HeteroData,
) -> Tensor:
    node_val_mask_list = list(graph.val_mask_dict.values())
    assert len(node_val_mask_list) == 1 

    return node_val_mask_list[0] 


def get_hetero_graph_target_node_test_mask(
    graph: HeteroData,
) -> Tensor:
    node_test_mask_list = list(graph.test_mask_dict.values())
    assert len(node_test_mask_list) == 1 

    return node_test_mask_list[0] 


def get_hetero_graph_adj_mat_dict(
    graph: HeteroData,
) -> dict[tuple[str, str, str], SparseTensor]:
    edge_index_dict = get_hetero_graph_edge_index_dict(graph)
    num_nodes_dict = get_hetero_graph_num_nodes_dict(graph)
    adj_mat_dict = dict() 

    for edge_type, edge_index in edge_index_dict.items():
        src_ntype, _, tgt_ntype = edge_type 
        num_src_nodes = num_nodes_dict[src_ntype]
        num_tgt_nodes = num_nodes_dict[tgt_ntype]

        adj_mat = SparseTensor.from_edge_index(
            edge_index = edge_index,
            sparse_sizes = (num_src_nodes, num_tgt_nodes),
        )
        adj_mat_dict[edge_type] = adj_mat 

    return adj_mat_dict 


def get_hetero_graph_adj_mat_t_dict(
    graph: HeteroData,
) -> dict[tuple[str, str, str], SparseTensor]:
    return {
        edge_type: adj_mat.t() 
        for edge_type, adj_mat in get_hetero_graph_adj_mat_dict(graph).items()
    }


def get_hetero_graph_short_etype_2_etype(
    graph: HeteroData,
) -> dict[str, tuple[str, str, str]]:
    edge_index_2d_dict = get_hetero_graph_edge_index_dict(graph)

    short_etype_2_etype = {
        edge_type[1]: edge_type
        for edge_type in edge_index_2d_dict.keys()
    }
    assert len(short_etype_2_etype) == len(edge_index_2d_dict) 

    return short_etype_2_etype 


def short_metapath_list_2_full_metapath_list(
    graph: HeteroData, 
    short_metapath_list: list[list[str]],
) -> list[list[tuple[str, str, str]]]:
    edge_type_set = get_hetero_graph_edge_type_set(graph)

    short_edge_type_2_edge_type = {
        edge_type[1]: edge_type 
        for edge_type in edge_type_set
    }
    assert len(short_edge_type_2_edge_type) == len(edge_type_set) 

    full_metapath_list = [
        [
            short_edge_type_2_edge_type[short_edge_type] 
            for short_edge_type in metapath 
        ]
        for metapath in short_metapath_list
    ]

    return full_metapath_list 

import torch
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_remaining_self_loops
from torch_sparse import matmul
import random 
from tqdm.auto import tqdm 
from typing import Optional


def extract_metapath_subgraphs(
    graph: HeteroData,
    metapath_list: list[list[str]],
) -> list[Tensor]:
    edge_index_2d_dict = get_hetero_graph_edge_index_dict(graph)
    num_nodes_dict = get_hetero_graph_num_nodes_dict(graph)

    short_etype_2_etype = {
        edge_type[1]: edge_type
        for edge_type in edge_index_2d_dict.keys()
    }
    assert len(short_etype_2_etype) == len(edge_index_2d_dict) 

    subgraph_edge_index_2d_list = []
    
    for metapath in metapath_list:
        adj_mat_prod_2d = None 
        
        for edge_type in metapath:
            edge_type = short_etype_2_etype[edge_type]
            src_ntype, _, tgt_ntype = edge_type
            num_src_nodes = num_nodes_dict[src_ntype]
            num_dst_nodes = num_nodes_dict[tgt_ntype]
            edge_index_2d = edge_index_2d_dict[edge_type]
            
            adj_mat_2d = torch.sparse_coo_tensor(
                indices = edge_index_2d,
                values = torch.ones(edge_index_2d.shape[1], device=edge_index_2d.device),
                size = (num_src_nodes, num_dst_nodes),
            )
            assert adj_mat_2d.shape == (num_src_nodes, num_dst_nodes)
            
            if adj_mat_prod_2d is None:
                adj_mat_prod_2d = adj_mat_2d
            else:
                adj_mat_prod_2d = adj_mat_prod_2d @ adj_mat_2d
                
        assert adj_mat_prod_2d is not None 
        adj_mat_prod_2d = adj_mat_prod_2d.coalesce()
        
        edge_index_2d = adj_mat_prod_2d.indices() 
        assert edge_index_2d.ndim == 2 and edge_index_2d.shape[0] == 2 
        
        subgraph_edge_index_2d_list.append(edge_index_2d)
    
    return subgraph_edge_index_2d_list


def propagate_along_metapaths(
    graph: HeteroData,
    metapath_list: list[list[str]],
    normalize_adj_mat: bool = True, 
) -> Tensor:
    adj_mat_t_dict = get_hetero_graph_adj_mat_t_dict(graph)
    edge_type_set = get_hetero_graph_edge_type_set(graph)
    node_feat_2d_dict = get_hetero_graph_node_feat_dict(graph)
    num_nodes_dict = get_hetero_graph_num_nodes_dict(graph)
    node_feat_dim = get_hetero_graph_target_node_feat_dim(graph)
    target_num_nodes = get_hetero_graph_target_num_nodes(graph)
    target_node_feat_2d = get_hetero_graph_target_node_feat(graph)

    short_etype_2_etype = {
        edge_type[1]: edge_type
        for edge_type in edge_type_set
    }
    assert len(short_etype_2_etype) == len(edge_type_set) 
    
    output_2d_list = [] 
    
    for metapath in metapath_list:
        if not metapath:
            output_2d_list.append(target_node_feat_2d)
            continue

        start_node_type = short_etype_2_etype[metapath[0]][0]
        propagated_node_feat_2d = node_feat_2d_dict[start_node_type]
        
        for edge_type in metapath:
            edge_type = short_etype_2_etype[edge_type]
            src_ntype, _, tgt_ntype = edge_type
            num_src_nodes = num_nodes_dict[src_ntype]
            num_dst_nodes = num_nodes_dict[tgt_ntype] 
            adj_mat_t = adj_mat_t_dict[edge_type]

            assert propagated_node_feat_2d.shape == (num_src_nodes, node_feat_dim)
            assert adj_mat_t.sparse_sizes() == (num_dst_nodes, num_src_nodes)

            if normalize_adj_mat:
                reduce = 'mean'
            else:
                reduce = 'sum'

            propagated_node_feat_2d = matmul(adj_mat_t, propagated_node_feat_2d, reduce=reduce)
            assert propagated_node_feat_2d.shape == (num_dst_nodes, node_feat_dim)
            
        output_2d_list.append(propagated_node_feat_2d)
        
    output_3d = torch.stack(output_2d_list, dim=1)
    assert output_3d.shape == (target_num_nodes, len(metapath_list), node_feat_dim)
    
    return output_3d 


def metapath_random_walk(
    edge_index_2d_dict: dict[tuple[str, str, str], Tensor], 
    num_nodes_dict: dict[str, int],
    metapath: list[str],
    sample_cnt: int,
) -> list[list[int]]:
    adj_list_dict = dict() 

    for edge_type, edge_index_2d in edge_index_2d_dict.items():
        src_node_type, _, dst_node_type = edge_type
        num_src_nodes = num_nodes_dict[src_node_type]
        num_dst_nodes = num_nodes_dict[dst_node_type]

        adj_list = [ [] for _ in range(num_src_nodes) ]

        for src_node_id, dst_node_id in edge_index_2d.t().tolist():
            adj_list[src_node_id].append(dst_node_id) 

        adj_list_dict[edge_type] = adj_list

    short_edge_type_2_edge_type = {
        edge_type[1]: edge_type 
        for edge_type in edge_index_2d_dict.keys()
    }
    assert len(short_edge_type_2_edge_type) == len(edge_index_2d_dict)

    start_node_type = short_edge_type_2_edge_type[metapath[0]][0]
    start_num_nodes = num_nodes_dict[start_node_type]

    path_list = [
        [i]
        for i in range(start_num_nodes)
        for _ in range(sample_cnt)
    ]
    assert len(path_list) == start_num_nodes * sample_cnt   

    for short_edge_type in tqdm(metapath, desc='metapath_random_walk()'):
        edge_type = short_edge_type_2_edge_type[short_edge_type]
        adj_list = adj_list_dict[edge_type] 

        new_path_list = [] 

        for path in path_list:
            node_id = path[-1] 
            neighbor_id_list = adj_list[node_id] 

            if neighbor_id_list:
                neighbor_id = random.choice(neighbor_id_list)
                path.append(neighbor_id)
                new_path_list.append(path)

        path_list = new_path_list

    return path_list 


def remove_metapath_path(
    edge_index_2d_dict: dict[tuple[str, str, str], Tensor],
    metapath: list[str],
    path_list: list[list[int]],
) -> dict[tuple[str, str, str], Tensor]:
    device = next(iter(edge_index_2d_dict.values())).device

    short_edge_type_2_edge_type = {
        edge_type[1]: edge_type 
        for edge_type in edge_index_2d_dict.keys()
    }
    assert len(short_edge_type_2_edge_type) == len(edge_index_2d_dict)

    removed_edge_index_2d_dict = edge_index_2d_dict.copy() 

    for i, short_edge_type in enumerate(metapath):
        edge_type = short_edge_type_2_edge_type[short_edge_type]
        edge_index_2d = edge_index_2d_dict[edge_type]

        edge_set = {
            (src_nid, dst_nid)
            for src_nid, dst_nid in edge_index_2d.t().tolist()
        }

        remove_edge_set = {
            (path[i], path[i + 1])
            for path in path_list
        }

        edge_set -= remove_edge_set

        removed_edge_index_2d = torch.tensor(list(edge_set), dtype=torch.int64, device=device).T.reshape(2, len(edge_set))
        removed_edge_index_2d_dict[edge_type] = removed_edge_index_2d

    return removed_edge_index_2d_dict

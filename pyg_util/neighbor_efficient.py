import torch 
from torch import Tensor 
from torch_sparse import SparseTensor
from typing import Optional 


def sample_neighbors(
    adj_mat_2d: SparseTensor, 
    start_node_1d: Tensor,
    num_neighbors: int,
) -> Tensor:
    num_src_nodes, num_dest_nodes = adj_mat_2d.sparse_sizes() 
    num_start_nodes, = start_node_1d.shape 
    num_edges = adj_mat_2d.nnz()
    device = adj_mat_2d.device()

    rowptr_1d, col_1d, _ = adj_mat_2d.csr()
    assert rowptr_1d.shape == (num_src_nodes + 1,)
    assert col_1d.shape == (num_edges,)

    rowcount_1d = adj_mat_2d.storage.rowcount()
    assert rowcount_1d.shape == (num_src_nodes,)

    rowcount_1d = rowcount_1d[start_node_1d]
    rowptr_1d = rowptr_1d[start_node_1d]
    assert rowcount_1d.shape == rowptr_1d.shape == (num_start_nodes,)

    # 确保每个起始节点都有邻居，否则采样会出错
    assert torch.all(rowcount_1d > 0)

    neighbor_eid_2d = torch.rand(num_start_nodes, num_neighbors, device=device)
    neighbor_eid_2d.mul_( rowcount_1d.to(torch.float32).reshape(num_start_nodes, 1) )
    neighbor_eid_2d = neighbor_eid_2d.to(torch.int64)
    neighbor_eid_2d.add_( rowptr_1d.reshape(num_start_nodes, 1) )
    assert neighbor_eid_2d.shape == (num_start_nodes, num_neighbors)

    neighbor_2d = col_1d[neighbor_eid_2d]
    assert neighbor_2d.shape == (num_start_nodes, num_neighbors)

    return neighbor_2d 


def metapath_random_walk(
    edge_index_2d_dict: dict[tuple[str, str, str], Tensor], 
    num_nodes_dict: dict[str, int],
    metapath: list[str],
    num_neighbors: int,
):
    device = next(iter(edge_index_2d_dict.values())).device

    adj_mat_2d_dict = {
        edge_type: SparseTensor.from_edge_index(
            edge_index = edge_index_2d, 
            sparse_sizes = (num_nodes_dict[edge_type[0]], num_nodes_dict[edge_type[-1]]),
        )
        for edge_type, edge_index_2d in edge_index_2d_dict.items()
    }

    short_edge_type_2_edge_type = {
        edge_type[1]: edge_type 
        for edge_type in edge_index_2d_dict.keys()
    }
    assert len(short_edge_type_2_edge_type) == len(edge_index_2d_dict)

    start_node_type = short_edge_type_2_edge_type[metapath[0]][0]
    start_node_1d = torch.arange(num_nodes_dict[start_node_type], device=device)
    sampled_edge_index_2d_list = [] 

    for short_edge_type in metapath:
        edge_type = short_edge_type_2_edge_type[short_edge_type] 
        src_node_type, _, dest_node_type = edge_type 
        adj_mat_2d = adj_mat_2d_dict[edge_type]

        dest_node_1d = sample_neighbors(
            adj_mat_2d = adj_mat_2d,
            start_node_1d = start_node_1d,
            num_neighbors = num_neighbors,
        )



def random_walk(
    edge_index_2d: Tensor,
    num_nodes: int,
    start_node_1d: Tensor,
    walk_length: int,
    p: float = 1.,
    q: float = 1.,
) -> tuple[Tensor, Tensor]:
    raise NotImplementedError

    if coalesced:
        perm = torch.argsort(row * num_nodes + col)
        row, col = row[perm], col[perm]

    deg = row.new_zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones_like(row))
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    node_seq, edge_seq = torch.ops.torch_cluster.random_walk(
        rowptr, col, start_node_1d, walk_length, p, q)

    if return_edge_indices:
        return node_seq, edge_seq

    return node_seq

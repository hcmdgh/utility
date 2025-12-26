import torch 
from torch import Tensor 
from torch_sparse import SparseTensor


def edge_index_to_rowptr_col(
    edge_index_2d: Tensor,
    num_src_nodes: int,
    num_dest_nodes: int,
) -> tuple[Tensor, Tensor]:
    num_edges = edge_index_2d.shape[1]
    assert edge_index_2d.shape == (2, num_edges)

    adj_mat_2d = SparseTensor(
        row = edge_index_2d[0], 
        col = edge_index_2d[1], 
        sparse_sizes = (num_src_nodes, num_dest_nodes),
    )
    
    rowptr_1d, col_1d, _ = adj_mat_2d.csr()
    assert rowptr_1d.shape == (num_src_nodes + 1,)
    assert col_1d.shape == (num_edges,)
    
    return rowptr_1d, col_1d


def row_normalize_adj_mat(
    adj_mat: SparseTensor,
) -> SparseTensor:
    device = adj_mat.device() 
    n, m = adj_mat.sparse_sizes()
    nnz = adj_mat.nnz()

    degree_1d = adj_mat.sum(dim=1)
    assert degree_1d.shape == (n,)

    inv_degree_1d = degree_1d.pow(-1)
    torch.nan_to_num_(inv_degree_1d, nan=0., posinf=0., neginf=0.)
    assert inv_degree_1d.shape == (n,)

    row_index_1d = adj_mat.storage.row()
    assert row_index_1d.shape == (nnz,) 

    value_1d = adj_mat.storage.value()
    if value_1d is None:
        value_1d = torch.ones(nnz, dtype=torch.float32, device=device)
    assert value_1d.shape == (nnz,)

    value_1d = value_1d * inv_degree_1d[row_index_1d]
    
    adj_mat = adj_mat.set_value(value_1d, layout='coo')

    return adj_mat 

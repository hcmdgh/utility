import torch 
from torch import Tensor 
import torch_scatter 


def sort_edge_index(
    edge_index: Tensor, 
    num_nodes: int | tuple[int, int],
    by: str,
) -> tuple[Tensor, Tensor]:
    num_edges = edge_index.shape[1]
    assert edge_index.shape == (2, num_edges)

    if isinstance(num_nodes, int):
        max_num_nodes = num_nodes
    elif isinstance(num_nodes, tuple):
        max_num_nodes = max(num_nodes)
    else:
        raise TypeError

    row = edge_index[0]
    col = edge_index[1]
    assert row.shape == col.shape == (num_edges,)
    
    if by == 'row':
        idx = row * max_num_nodes + col
    elif by == 'col':
        idx = col * max_num_nodes + row
    else:
        raise ValueError 
    
    perm = idx.argsort()
    assert perm.shape == (num_edges,)
    
    return edge_index[:, perm], perm 


def edge_index_to_row_col_colptr(
    edge_index: Tensor,
    num_nodes: int | tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    num_edges = edge_index.shape[1]
    assert edge_index.shape == (2, num_edges)

    if isinstance(num_nodes, int):
        num_nodes = (num_nodes, num_nodes)
    elif isinstance(num_nodes, tuple):
        pass 
    else:
        raise TypeError

    edge_index, perm = sort_edge_index(
        edge_index = edge_index, 
        num_nodes = num_nodes, 
        by = 'col',
    )
    row = edge_index[0]
    col = edge_index[1]
    assert row.shape == col.shape == (num_edges,)

    degree = torch.bincount(col, minlength=num_nodes[1])
    assert degree.shape == (num_nodes[1],)

    cumsum_degree = torch.cumsum(degree, dim=0)
    assert cumsum_degree.shape == (num_nodes[1],)

    col_ptr = torch.cat(
        [
            torch.zeros(1, dtype=cumsum_degree.dtype, device=cumsum_degree.device), 
            cumsum_degree,
        ],
        dim = 0,
    )
    assert col_ptr.shape == (num_nodes[1] + 1,)

    return row, col, col_ptr, perm


def edge_index_to_row_col_rowptr(
    edge_index: Tensor,
    num_nodes: int | tuple[int, int],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    num_edges = edge_index.shape[1]
    assert edge_index.shape == (2, num_edges)

    if isinstance(num_nodes, int):
        num_nodes = (num_nodes, num_nodes)
    elif isinstance(num_nodes, tuple):
        pass 
    else:
        raise TypeError

    edge_index, perm = sort_edge_index(
        edge_index = edge_index, 
        num_nodes = num_nodes, 
        by = 'row',
    )
    row = edge_index[0]
    col = edge_index[1]
    assert row.shape == col.shape == (num_edges,)

    degree = torch.bincount(row, minlength=num_nodes[0])
    assert degree.shape == (num_nodes[0],)

    cumsum_degree = torch.cumsum(degree, dim=0)
    assert cumsum_degree.shape == (num_nodes[0],)

    row_ptr = torch.cat(
        [
            torch.zeros(1, dtype=cumsum_degree.dtype, device=cumsum_degree.device), 
            cumsum_degree,
        ],
        dim = 0,
    )
    assert row_ptr.shape == (num_nodes[0] + 1,)

    return row, col, row_ptr, perm 
    

def segment_csr(
    src: Tensor,
    ptr: Tensor,
    reduce: str,
) -> Tensor:
    num_edges, feat_dim = src.shape
    num_nodes, = ptr.shape 
    num_nodes -= 1 

    output = torch_scatter.segment_csr(
        src = src,
        indptr = ptr,
        reduce = reduce,
    )
    assert output.shape == (num_nodes, feat_dim)

    return output 


def segment_csr_add(
    src: Tensor,
    ptr: Tensor,
) -> Tensor:
    return segment_csr(
        src = src, 
        ptr = ptr, 
        reduce = 'sum',
    )


def softmax_csr(
    src: Tensor,
    index: Tensor,
    ptr: Tensor,
) -> Tensor:
    num_edges, feat_dim = src.shape
    num_nodes, = ptr.shape 
    num_nodes -= 1 
    assert index.shape == (num_edges,)

    src_max = torch_scatter.segment_csr(
        src = src, 
        indptr = ptr, 
        reduce = 'max',
    )
    assert src_max.shape == (num_nodes, feat_dim)
    
    output = (src - src_max[index]).exp()
    assert output.shape == (num_edges, feat_dim)
    
    output_sum = torch_scatter.segment_csr(
        src = output, 
        indptr = ptr, 
        reduce = 'sum',
    )
    assert output_sum.shape == (num_nodes, feat_dim)
    
    output = output / (output_sum[index] + 1e-16)

    return output

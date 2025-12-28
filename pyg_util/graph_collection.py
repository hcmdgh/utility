import torch
from torch import Tensor
from torch_scatter import segment_csr
from typing import Literal


def global_pooling(
    node_embedding_2d: Tensor,
    graph_ptr_1d: Tensor,
    reduce: str = 'mean',
) -> Tensor:
    num_nodes, node_embedding_dim = node_embedding_2d.shape
    num_graphs, = graph_ptr_1d.shape
    num_graphs -= 1
    assert int(graph_ptr_1d[0]) == 0
    assert int(graph_ptr_1d[-1]) == num_nodes

    if reduce == 'none':
        return node_embedding_2d
    elif reduce in ['sum', 'mean', 'max', 'min']:
        out = segment_csr(
            src = node_embedding_2d,
            indptr = graph_ptr_1d,
            reduce = reduce,
        )
        assert out.shape == (num_graphs, node_embedding_dim)

        return out
    else:
        raise ValueError

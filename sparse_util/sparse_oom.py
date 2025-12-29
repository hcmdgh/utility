import torch 
from torch import Tensor 
import warnings
from typing import Any, Optional 

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta state.*")


class SparseTensor:
    def __init__(
        self,
        data: Tensor,
    ):
        self.data = data 
        
        self.coalesce_()

    @classmethod
    def from_edge_index(
        cls,
        edge_index: Tensor,
        num_nodes: int | tuple[int, int] | list[int],
        edge_weight: Optional[Tensor] = None,
    ) -> 'SparseTensor':
        device = edge_index.device
        num_edges = edge_index.shape[1]
        assert edge_index.shape == (2, num_edges)

        if isinstance(num_nodes, int):
            num_nodes = (num_nodes, num_nodes)
        elif isinstance(num_nodes, (list, tuple)):
            n, m = num_nodes 
            num_nodes = (n, m)
        else:
            raise TypeError 
        
        if edge_weight is not None:
            assert edge_weight.shape == (num_edges,)
        else:
            edge_weight = torch.ones(num_edges, dtype=torch.float32, device=device)
        
        data = torch.sparse_coo_tensor(
            indices = edge_index,
            values = edge_weight,
            size = num_nodes,
        )

        return cls(data=data)

    @classmethod
    def from_pyg_data(
        cls,
        data: Any,
    ) -> 'SparseTensor':
        return cls.from_edge_index(
            edge_index = data['edge_index'],
            num_nodes = data['num_nodes'],
        )
    
    @classmethod 
    def from_dense_tensor(
        cls,
        tensor: Tensor,
    ) -> 'SparseTensor':
        return cls(data=tensor.to_sparse_coo())
    
    @classmethod 
    def diagonal(
        cls,
        diagonal: Tensor,
    ) -> 'SparseTensor':
        device = diagonal.device 
        num_nodes, = diagonal.shape 

        edge_index = torch.arange(num_nodes, dtype=torch.int64, device=device).reshape(1, num_nodes).repeat(2, 1)
        assert edge_index.shape == (2, num_nodes) 

        return cls.from_edge_index(
            edge_index = edge_index,
            num_nodes = num_nodes,
            edge_weight = diagonal,
        )
    
    @classmethod 
    def eye(
        cls,
        num_nodes: int,
        device: Any = 'cpu',
    ) -> 'SparseTensor':
        diagonal = torch.ones(num_nodes, dtype=torch.float32, device=device)

        return cls.diagonal(diagonal=diagonal)

    def coalesce_(self) -> 'SparseTensor':
        assert self.data.layout == torch.sparse_coo
        assert self.data.ndim == 2 
        
        self.data = self.data.coalesce()
        assert self.data.indices().dtype == torch.int64
        assert self.data.values().dtype == torch.float32

        return self 
    
    @property 
    def shape(self) -> tuple[int, int]:
        n, m = self.data.shape
        return n, m

    num_nodes = shape 

    @property 
    def nnz(self) -> int:
        return self.data._nnz() 
    
    num_edges = nnz 

    @property 
    def density(self) -> float:
        return self.nnz / (self.shape[0] * self.shape[1])
    
    @property
    def edge_index(self) -> Tensor:
        edge_index = self.data.indices()
        assert edge_index.shape == (2, self.nnz)
        
        return edge_index 

    @property 
    def row(self) -> Tensor:
        return self.edge_index[0]
    
    @property 
    def col(self) -> Tensor:
        return self.edge_index[1]
    
    @property 
    def value(self) -> Tensor:
        value = self.data.values()
        assert value.shape == (self.nnz,)

        return value 

    def copy(self) -> 'SparseTensor':
        return SparseTensor(data=self.data.clone())
    
    def to_dense(self) -> Tensor:
        return self.data.to_dense()
    
    def sparse_matmul(
        self,
        other: 'SparseTensor',
    ) -> 'SparseTensor':
        new_data = self.data @ other.data
        assert new_data.layout == torch.sparse_coo

        return SparseTensor(data=new_data)
    
    def dense_matmul(
        self,
        other: Tensor,
    ) -> Tensor:
        assert other.layout == torch.strided

        output = self.data @ other
        assert output.layout == torch.strided 

        return output 
    
    def in_degree(self) -> Tensor:
        degree = self.data.sum(dim=0).to_dense()
        assert degree.shape == (self.shape[1],)

        return degree 
    
    def out_degree(self) -> Tensor:
        degree = self.data.sum(dim=1).to_dense()
        assert degree.shape == (self.shape[0],)

        return degree 
    
    def mul_vec(
        self,
        other: Tensor,
    ) -> 'SparseTensor':
        assert other.layout == torch.strided
        assert other.shape == (1, self.shape[1]) or other.shape == (self.shape[0], 1)

        new_data = self.data * other

        return SparseTensor(data=new_data)
    
    def row_normalize(self) -> 'SparseTensor':
        degree = self.out_degree().reshape(self.shape[0], 1)
        degree_inv = degree.pow(-1) 
        torch.nan_to_num_(degree_inv, nan=0., posinf=0., neginf=0.)

        return self.mul_vec(degree_inv)

    def col_normalize(self) -> 'SparseTensor':
        degree = self.in_degree().reshape(1, self.shape[1])
        degree_inv = degree.pow(-1) 
        torch.nan_to_num_(degree_inv, nan=0., posinf=0., neginf=0.)

        return self.mul_vec(degree_inv)
    
    def gcn_normalize(self) -> 'SparseTensor':
        in_degree = self.in_degree().reshape(1, self.shape[1])
        out_degree = self.out_degree().reshape(self.shape[0], 1)
        in_degree_inv_sqrt = in_degree.pow(-0.5) 
        out_degree_inv_sqrt = out_degree.pow(-0.5) 
        torch.nan_to_num_(in_degree_inv_sqrt, nan=0., posinf=0., neginf=0.)
        torch.nan_to_num_(out_degree_inv_sqrt, nan=0., posinf=0., neginf=0.)

        return self.mul_vec(in_degree_inv_sqrt).mul_vec(out_degree_inv_sqrt)
    
    @classmethod 
    def stack(
        cls,
        sparse_tensor_list: list['SparseTensor'],
    ) -> tuple[Tensor, Tensor]:
        _sum = sparse_tensor_list[0].data 

        for sparse_tensor in sparse_tensor_list[1:]:
            _sum = _sum + sparse_tensor.data
            
        zero = (_sum * 0.).coalesce()
        edge_index = zero.indices()
        num_edges = edge_index.shape[1]

        value_list = [] 

        for sparse_tensor in sparse_tensor_list:
            sparse_tensor = (sparse_tensor.data + zero).coalesce()
            _edge_index = sparse_tensor.indices()
            value = sparse_tensor.values()
            assert torch.all(_edge_index == edge_index)

            value_list.append(value) 

        value = torch.stack(value_list, dim=-1)
        assert value.shape == (num_edges, len(sparse_tensor_list))

        return edge_index, value 

    def __str__(self) -> str:
        return self.data.__str__()
    
    def __repr__(self) -> str:
        return self.data.__repr__()

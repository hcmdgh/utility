import torch 
from torch import Tensor 
import torch_sparse 
import warnings
from typing import Any, Optional 

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support is in beta state.*")


class SparseTensor:
    def __init__(
        self,
        data: torch_sparse.SparseTensor,
    ):
        assert isinstance(data, torch_sparse.SparseTensor)
        self.data = data 

        assert len(self.data.sizes()) == 2 
        row, col, value = self.data.coo()
        assert row.dtype == col.dtype == torch.int64  
        assert value is not None and value.dtype == torch.float32 and value.ndim == 1 
        
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
        
        data = torch_sparse.SparseTensor.from_edge_index(
            edge_index = edge_index,
            edge_attr = edge_weight,
            sparse_sizes = num_nodes,
        )

        return cls(data=data)

    @classmethod
    def from_pyg_data(
        cls,
        data: Any,
    ) -> 'SparseTensor':
        if 'num_nodes' in data:
            num_nodes = int(data['num_nodes'])
        elif 'x' in data:
            num_nodes = len(data['x']) 
        else:
            raise ValueError

        return cls.from_edge_index(
            edge_index = data['edge_index'],
            num_nodes = num_nodes,
        )
    
    @classmethod 
    def from_dense_tensor(
        cls,
        tensor: Tensor,
    ) -> 'SparseTensor':
        return cls(data=torch_sparse.SparseTensor.from_dense(tensor))
    
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
    
    @classmethod
    def eye_like(
        cls,
        sparse_tensor: 'SparseTensor',
    ) -> 'SparseTensor':
        assert sparse_tensor.is_square

        return cls.eye(
            num_nodes = sparse_tensor.shape[0], 
            device = sparse_tensor.device,
        )
    
    @classmethod
    def zeros_like(
        cls,
        sparse_tensor: 'SparseTensor',
    ) -> 'SparseTensor':
        return cls.from_edge_index(
            edge_index = sparse_tensor.edge_index,
            edge_weight = torch.zeros_like(sparse_tensor.value), 
            num_nodes = sparse_tensor.num_nodes,
        )

    def coalesce_(self) -> 'SparseTensor':
        self.data = self.data.coalesce()

        return self 
    
    @property 
    def shape(self) -> tuple[int, int]:
        n, m = self.data.sparse_sizes()
        return n, m

    num_nodes = shape 

    @property 
    def is_square(self) -> bool:
        return self.shape[0] == self.shape[1]

    @property 
    def nnz(self) -> int:
        return self.data.nnz()
    
    num_edges = nnz 

    @property 
    def density(self) -> float:
        return self.nnz / (self.shape[0] * self.shape[1])

    @property 
    def row(self) -> Tensor:
        row = self.data.storage.row() 
        assert row.shape == (self.nnz,)

        return row 
    
    @property 
    def col(self) -> Tensor:
        col = self.data.storage.col() 
        assert col.shape == (self.nnz,)

        return col 
    
    @property
    def edge_index(self) -> Tensor:
        edge_index = torch.stack([self.row, self.col], dim=0)
        assert edge_index.shape == (2, self.nnz)
        
        return edge_index 
    
    @property 
    def value(self) -> Tensor:
        value = self.data.storage.value()
        assert value is not None 
        assert value.shape == (self.nnz,)

        return value 
    
    edge_weight = value
    
    @property 
    def device(self) -> torch.device:
        return self.data.device()

    def copy(self) -> 'SparseTensor':
        return SparseTensor(data=self.data.clone())
    
    def to_dense(self) -> Tensor:
        return self.data.to_dense()
    
    def matmul(
        self,
        other: Any,
    ) -> Any:
        if isinstance(other, SparseTensor):
            result = self.data @ other.data

            return SparseTensor(data=result)
        elif isinstance(other, Tensor):
            assert other.layout == torch.strided
            assert other.ndim == 2 

            result = self.data @ other 
            assert result.layout == torch.strided

            return result 
        else:
            raise TypeError
    
    def add(
        self,
        other: 'SparseTensor',
    ) -> 'SparseTensor':
        return SparseTensor(data=self.data + other.data)
    
    def mul(
        self,
        other: Any,
    ) -> 'SparseTensor':
        if isinstance(other, SparseTensor):
            result = self.data * other.data
        elif isinstance(other, Tensor):
            assert other.layout == torch.strided
            assert other.shape == (1, self.shape[1]) or other.shape == (self.shape[0], 1)

            result = self.data * other
        elif isinstance(other, float):
            result = self.data * other
        else:
            raise TypeError 

        return SparseTensor(data=result)
    
    def in_degree(self) -> Tensor:
        degree = self.data.sum(dim=0) 
        assert degree.layout == torch.strided
        assert degree.shape == (self.shape[1],)

        return degree 
    
    def out_degree(self) -> Tensor:
        degree = self.data.sum(dim=1)
        assert degree.layout == torch.strided
        assert degree.shape == (self.shape[0],)

        return degree 
    
    def row_normalize(self) -> 'SparseTensor':
        relu = self.relu()
        degree = relu.out_degree().reshape(relu.shape[0], 1)
        degree_inv = degree.pow(-1) 
        torch.nan_to_num_(degree_inv, nan=0., posinf=0., neginf=0.)

        return relu.mul(degree_inv)

    def col_normalize(self) -> 'SparseTensor':
        relu = self.relu()
        degree = relu.in_degree().reshape(1, relu.shape[1])
        degree_inv = degree.pow(-1) 
        torch.nan_to_num_(degree_inv, nan=0., posinf=0., neginf=0.)

        return relu.mul(degree_inv)
    
    def gcn_normalize(self) -> 'SparseTensor':
        in_degree = self.in_degree().reshape(1, self.shape[1])
        out_degree = self.out_degree().reshape(self.shape[0], 1)
        in_degree_inv_sqrt = in_degree.pow(-0.5) 
        out_degree_inv_sqrt = out_degree.pow(-0.5) 
        torch.nan_to_num_(in_degree_inv_sqrt, nan=0., posinf=0., neginf=0.)
        torch.nan_to_num_(out_degree_inv_sqrt, nan=0., posinf=0., neginf=0.)

        return self.mul(in_degree_inv_sqrt).mul(out_degree_inv_sqrt)
    
    def exp(self) -> 'SparseTensor':
        data = self.data 
        value = self.value 
        value = torch.exp(value)
        data = data.set_value(value, layout='coo')

        return SparseTensor(data=data)
    
    def relu(self) -> 'SparseTensor':
        data = self.data 
        value = self.value 
        value = torch.relu(value)
        data = data.set_value(value, layout='coo')

        return SparseTensor(data=data)
    
    def row_softmax(self) -> 'SparseTensor':
        return self.exp().row_normalize()
    
    def col_softmax(self) -> 'SparseTensor':
        return self.exp().col_normalize()
    
    @classmethod 
    def sum(
        cls,
        sparse_tensor_list: list['SparseTensor'],
    ) -> 'SparseTensor':
        sum_data = sparse_tensor_list[0].data

        for sparse_tensor in sparse_tensor_list[1:]:
           sum_data = sum_data + sparse_tensor.data

        return cls(data=sum_data)

    @classmethod 
    def stack(
        cls,
        sparse_tensor_list: list['SparseTensor'],
    ) -> tuple[Tensor, Tensor]:
        sum = cls.sum(sparse_tensor_list)
        zero = cls.zeros_like(sum)  
        edge_index = zero.edge_index
        num_edges = zero.num_edges

        edge_weight_list = [] 

        for sparse_tensor in sparse_tensor_list:
            sparse_tensor = sparse_tensor.add(zero)
            edge_weight = sparse_tensor.edge_weight
            assert torch.all(sparse_tensor.edge_index == edge_index)

            edge_weight_list.append(edge_weight) 

        edge_weight_2d = torch.stack(edge_weight_list, dim=-1)
        assert edge_weight_2d.shape == (num_edges, len(sparse_tensor_list))

        return edge_index, edge_weight_2d  

    def __str__(self) -> str:
        return self.data.__str__()
    
    def __repr__(self) -> str:
        return self.data.__repr__()

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor 
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm.auto import tqdm 
from typing import Any 


class BatchNorm(nn.Module):
    def __init__(
        self,
        in_dim: int,
        enable: bool = True,
    ):
        super().__init__()
        
        self.in_dim = in_dim 
        self.enable = enable 
        
        if enable:
            self.bn = nn.BatchNorm1d(in_dim)
        else:
            self.bn = nn.Identity()
            
    def forward(
        self, 
        input_2d: Tensor,
    ) -> Tensor:
        batch_size = len(input_2d)
        assert input_2d.shape == (batch_size, self.in_dim)
        
        out_2d = self.bn(input_2d)
        assert out_2d.shape == (batch_size, self.in_dim)
        
        return out_2d 
    

class Activation(nn.Module):
    def __init__(self, mode: str):
        super().__init__()

        if mode == 'relu':
            self.activation = nn.ReLU()
        elif mode == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif mode == 'tanh':
            self.activation = nn.Tanh()
        elif mode == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif mode == 'elu':
            self.activation = nn.ELU()
        elif mode == 'gelu':
            self.activation = nn.GELU()
        elif mode == 'prelu':
            self.activation = nn.PReLU()
        elif mode == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError

    def forward(self, input: Tensor) -> Tensor:
        return self.activation(input)


class Linear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        learnable: bool = True,
        batch_norm: bool = False,
        activation: str = 'none',
        dropout: float = 0.,
    ):
        super().__init__()
        
        self.in_dim = in_dim 
        self.out_dim = out_dim
        
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        
        self.batch_norm = BatchNorm(in_dim=out_dim, enable=batch_norm)
            
        self.activation = Activation(activation)
        
        self.dropout = nn.Dropout(dropout)
        
        if not learnable:
            for p in self.parameters():
                p.requires_grad = False
        
    def forward(self, input_2d: Tensor) -> Tensor:
        batch_size = len(input_2d)
        assert input_2d.shape == (batch_size, self.in_dim)
        
        output_2d = self.linear(input_2d)
        assert output_2d.shape == (batch_size, self.out_dim)

        output_2d = self.batch_norm(output_2d)
        output_2d = self.activation(output_2d)
        output_2d = self.dropout(output_2d)
        assert output_2d.shape == (batch_size, self.out_dim) 
        
        return output_2d
    

class NonLinear(nn.Module):
    def __init__(
        self,
        dim: int,
        batch_norm: bool = False,
        activation: str = 'relu',
        dropout: float = 0.,
    ):
        super().__init__()
        
        self.batch_norm = BatchNorm(in_dim=dim, enable=batch_norm)
        
        self.activation = Activation(activation)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_2d: Tensor,
    ) -> Tensor:
        output_2d = self.batch_norm(input_2d)
        output_2d = self.activation(output_2d)
        output_2d = self.dropout(output_2d)
        
        return output_2d 
    

class NonLinearWithResidual(nn.Module):
    def __init__(
        self,
        in_dim: int,
        residual_in_dim: int,
        batch_norm: bool = False,
        activation: str = 'relu',
        dropout: float = 0.,
        residual_type: str = 'none',
        residual_position: str = 'before',
    ):
        super().__init__()

        self.in_dim = in_dim
        self.residual_position = residual_position

        self.batch_norm = BatchNorm(in_dim=in_dim, enable=batch_norm)

        self.activation = Activation(activation)

        self.dropout = nn.Dropout(dropout)

        self.residual = Residual(
            in_dim = in_dim,
            residual_in_dim = residual_in_dim,
            mode = residual_type,
        )

    def forward(
        self,
        input: Tensor,
        residual_input: Tensor,
    ) -> Tensor:
        if self.residual_position == 'before':
            output = self.residual(input=input, residual_input=residual_input)
            output = self.batch_norm(output)
            output = self.activation(output)
            output = self.dropout(output)
        elif self.residual_position == 'after':
            output = self.batch_norm(input)
            output = self.activation(output)
            output = self.dropout(output)
            output = self.residual(input=output, residual_input=residual_input)
        else:
            raise ValueError

        return output
    

class Residual(nn.Module):
    def __init__(
        self,
        in_dim: int,
        residual_in_dim: int,
        mode: str,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.residual_in_dim = residual_in_dim
        self.mode = mode
        
        if mode == 'none':
            self.linear = None
        elif mode == 'add':
            assert residual_in_dim == in_dim
            self.linear = None
        elif mode == 'lin_add':
            self.linear = nn.Linear(residual_in_dim, in_dim)
        elif mode == 'concat_lin':
            self.linear = nn.Linear(in_dim + residual_in_dim, in_dim)
        elif mode == 'concat_lin_add':
            assert residual_in_dim == in_dim
            self.linear = nn.Linear(in_dim * 2, in_dim)
        else:
            raise ValueError
        
    def forward(
        self,
        input: Tensor,
        residual_input: Tensor,
    ) -> Tensor:
        batch_size = len(input) 
        assert input.shape == (batch_size, self.in_dim) 
        assert residual_input.shape == (batch_size, self.residual_in_dim)  
        
        if self.mode == 'none':
            return input
        elif self.mode == 'add':
            assert self.residual_in_dim == self.in_dim
            return input + residual_input
        elif self.mode == 'lin_add':
            assert self.linear
            return input + self.linear(residual_input)
        elif self.mode == 'concat_lin':
            assert self.linear
            return self.linear(torch.cat([input, residual_input], dim=-1))
        elif self.mode == 'concat_lin_add':
            assert self.linear
            assert self.residual_in_dim == self.in_dim
            return self.linear(torch.cat([input, residual_input], dim=-1)) + residual_input 
        else:
            raise ValueError
    

class MLP(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int = -1,
        out_dim: int,
        num_layers: int,
        bias: bool = True,
        is_last_layer: bool = True,
        batch_norm: bool = False,
        activation: str = 'relu',
        dropout: float = 0.,
    ):
        super().__init__()

        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.num_layers = num_layers

        if hidden_dim <= 0:
            self.hidden_dim = hidden_dim = (in_dim + out_dim) // 2 
        else:
            self.hidden_dim = hidden_dim

        if num_layers > 1:
            linear_list = [
                Linear(
                    in_dim = in_dim,
                    out_dim = hidden_dim,
                    bias = bias,
                    batch_norm = batch_norm,
                    activation = activation,
                    dropout = dropout,
                ),
                *[
                    Linear(
                        in_dim = hidden_dim,
                        out_dim = hidden_dim,
                        bias = bias,
                        batch_norm = batch_norm,
                        activation = activation,
                        dropout = dropout,
                    )
                    for _ in range(num_layers - 2)
                ],
                Linear(
                    in_dim = hidden_dim,
                    out_dim = out_dim,
                    bias = bias,
                    batch_norm = False if is_last_layer else batch_norm,
                    activation = 'none' if is_last_layer else activation,
                    dropout = 0. if is_last_layer else dropout,
                ),
            ]
        elif num_layers == 1:
            linear_list = [
                Linear(
                    in_dim = in_dim,
                    out_dim = out_dim,
                    bias = bias,
                    batch_norm = False if is_last_layer else batch_norm,
                    activation = 'none' if is_last_layer else activation,
                    dropout = 0. if is_last_layer else dropout,
                )
            ]
        else:
            raise ValueError 

        assert len(linear_list) == self.num_layers 

        self.sequential = nn.Sequential(*linear_list)

    def forward(
        self,
        input_2d: Tensor,
    ) -> Tensor:
        batch_size = len(input_2d)
        assert input_2d.shape == (batch_size, self.in_dim) 

        out_2d = self.sequential(input_2d) 
        assert out_2d.shape == (batch_size, self.out_dim) 

        return out_2d 


class LinearDict(nn.Module):
    def __init__(
        self,
        in_dim_dict: dict[str, int],
        out_dim: int,
        batch_norm: bool = False,
        activation: str = 'none',
        dropout: float = 0.,
    ):
        super().__init__()

        self.in_dim_dict = in_dim_dict 
        self.out_dim = out_dim 
        self.channel_set = set(in_dim_dict.keys())

        self.linear_dict = nn.ModuleDict({
            channel: Linear(
                in_dim = in_dim,
                out_dim = out_dim,
                batch_norm = batch_norm,
                activation = activation,
                dropout = dropout,
            )
            for channel, in_dim in in_dim_dict.items()
        })

    def __getitem__(self, key: str) -> Linear:
        return self.linear_dict[key]

    def forward(
        self,
        input_2d_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        output_2d_dict = dict() 

        for channel in self.channel_set:
            in_dim = self.in_dim_dict[channel]
            input_2d = input_2d_dict[channel]
            batch_size = len(input_2d)
            assert input_2d.shape == (batch_size, in_dim)

            output_2d = self.linear_dict[channel](input_2d)
            assert output_2d.shape == (batch_size, self.out_dim)

            output_2d_dict[channel] = output_2d 

        return output_2d_dict 


class NonlinearDict(nn.Module):
    def __init__(
        self,
        in_dim_dict: dict[str, int],
        batch_norm: bool = False,
        activation: str = 'none',
        dropout: float = 0.,
    ):
        super().__init__()

        self.in_dim_dict = in_dim_dict 
        self.channel_set = set(in_dim_dict.keys())

        self.batch_norm_dict = nn.ModuleDict({
            channel: BatchNorm(
                in_dim = in_dim,
                enable = batch_norm,
            )
            for channel, in_dim in in_dim_dict.items() 
        })

        self.activation = Activation(activation)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_2d_dict: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        output_2d_dict = dict() 

        for channel in self.channel_set:
            in_dim = self.in_dim_dict[channel]
            input_2d = input_2d_dict[channel]
            batch_size = len(input_2d)
            assert input_2d.shape == (batch_size, in_dim)

            output_2d = self.batch_norm_dict[channel](input_2d)
            output_2d = self.activation(output_2d)
            output_2d = self.dropout(output_2d)
            assert output_2d.shape == (batch_size, in_dim)

            output_2d_dict[channel] = output_2d 

        return output_2d_dict 


class Transformer(nn.Module):
    def __init__(
        self,
        sequence_len: int,
        embedding_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        layer_norm: bool,
        activation: str,
        dropout: float,
    ):
        super().__init__()
        
        self.sequence_len = sequence_len
        self.embedding_dim = embedding_dim
        
        if layer_norm:
            self.layer_norm = nn.LayerNorm(embedding_dim)
        else:
            self.layer_norm = None 
            
        self.encoder = nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = embedding_dim,
                nhead = num_heads,
                dim_feedforward = feedforward_dim,
                dropout = dropout,
                activation = Activation(activation),
                batch_first = True,
            ),
            num_layers = num_layers,
            norm = self.layer_norm,
        )
        
    def forward(
        self,
        input_3d: Tensor,
    ) -> Tensor:
        batch_size = len(input_3d)
        assert input_3d.shape == (batch_size, self.sequence_len, self.embedding_dim)
        
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_3d = self.encoder(input_3d)
            assert output_3d.shape == (batch_size, self.sequence_len, self.embedding_dim)
        
        return output_3d 


def run_linear_probe(
    train_feat_2d: Tensor, 
    train_label_1d: Tensor,
    val_feat_2d: Tensor,
    val_label_1d: Tensor,
    test_feat_2d: Tensor,
    test_label_1d: Tensor,
    use_ap: bool = False,
    lr: float = 0.001,
    num_epochs: int = 1000,
    early_stopping_patience: int = 30,
    use_tqdm: bool = False,
) -> dict[str, Any]:
    train_size = len(train_feat_2d)
    feat_dim = train_feat_2d.shape[-1]
    val_size = len(val_feat_2d)
    test_size = len(test_feat_2d)
    assert train_feat_2d.shape == (train_size, feat_dim)
    assert train_feat_2d.dtype == torch.float32
    assert train_label_1d.shape == (train_size,)
    assert train_label_1d.dtype == torch.int64 
    assert val_feat_2d.shape == (val_size, feat_dim)
    assert val_feat_2d.dtype == torch.float32
    assert val_label_1d.shape == (val_size,)
    assert val_label_1d.dtype == torch.int64 
    assert test_feat_2d.shape == (test_size, feat_dim)
    assert test_feat_2d.dtype == torch.float32
    assert test_label_1d.shape == (test_size,)
    assert test_label_1d.dtype == torch.int64 

    device = train_feat_2d.device 
    out_dim = int(torch.cat([train_label_1d, val_label_1d, test_label_1d]).max() + 1)
    
    model = nn.Linear(feat_dim, out_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if metric == 'acc':
        metric_func = compute_acc 
    elif metric == 'ap':
        metric_func = compute_ap 
    else:
        raise ValueError 
    
    best_epoch = -1 
    best_val_metric = -1.
    final_train_metric = -1.
    final_test_metric = -1. 
    patience_counter = 0 
    
    for epoch in tqdm(range(1, num_epochs + 1), desc='Linear Probing', disable=not use_tqdm):
        pred = model(train_feat_2d)
        
        loss = F.cross_entropy(input=pred, target=train_label_1d)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            logit_2d = model(val_feat_2d).detach() 
            
        val_metric = metric_func(
            logit_2d = logit_2d,
            label_1d = val_label_1d,
        )
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_epoch = epoch 
            patience_counter = 0

            with torch.no_grad():
                logit_2d = model(train_feat_2d).detach() 
            
            final_train_metric = metric_func(
                logit_2d = logit_2d,
                label_1d = train_label_1d,
            )
            
            with torch.no_grad():
                logit_2d = model(test_feat_2d).detach() 
            
            final_test_metric = metric_func(
                logit_2d = logit_2d,
                label_1d = test_label_1d,
            )
        else:
            patience_counter += 1 
            
            if patience_counter >= early_stopping_patience:
                break 
            
    return dict(
        best_epoch = best_epoch,
        final_train_metric = final_train_metric,
        best_val_metric = best_val_metric,
        final_test_metric = final_test_metric,
    )

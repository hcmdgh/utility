import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor 


def compute_cross_entropy_loss(
    input: Tensor,
    target: Tensor,
) -> Tensor:
    batch_size, num_classes = input.shape 
    assert target.shape == (batch_size,) 

    loss = F.cross_entropy(input=input, target=target)

    return loss 

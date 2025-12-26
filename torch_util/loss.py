import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch import Tensor 


def compute_cross_entropy_loss(
    logit_2d: Tensor,
    label_1d: Tensor,
) -> Tensor:
    batch_size, num_classes = logit_2d.shape 
    assert logit_2d.dtype == torch.float32 and label_1d.dtype == torch.int64 
    assert label_1d.shape == (batch_size,) 

    loss = F.cross_entropy(input=logit_2d, target=label_1d)

    return loss 

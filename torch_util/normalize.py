import torch 
from torch import Tensor 
from sklearn.preprocessing import StandardScaler


def normalize_feature(feat_2d: Tensor) -> Tensor:
    device = feat_2d.device 
    N, D = feat_2d.shape 
    assert feat_2d.dtype == torch.float32 
    
    scaler = StandardScaler()
    
    output_2d = scaler.fit_transform(feat_2d.detach().cpu().numpy())
    output_2d = torch.tensor(output_2d, dtype=torch.float32, device=device)
    assert output_2d.shape == (N, D)
    
    return output_2d 

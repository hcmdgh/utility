import torch 
import torch.nn as nn
from torch import Tensor 
from typing import Optional, Any 
from tqdm import tqdm  

from .model import MLP 
from .supervised_recorder import SupervisedRecorder 
from .loss import compute_cross_entropy_loss 


def run_linear_probe(
    train_embedding_2d: Tensor, 
    train_label_1d: Tensor,
    val_embedding_2d: Tensor,
    val_label_1d: Tensor,
    test_embedding_2d: Tensor,
    test_label_1d: Tensor,
    main_metric: str,
    metrics: list[str],
    num_layers: int,
    lr: float = 0.001,
    num_epochs: int = 1000,
    early_stopping_patience: int = 30,
    use_tqdm: bool = False,
) -> dict[str, Any]:
    train_size = len(train_embedding_2d)
    feat_dim = train_embedding_2d.shape[-1]
    val_size = len(val_embedding_2d)
    test_size = len(test_embedding_2d)
    assert train_embedding_2d.shape == (train_size, feat_dim)
    assert train_embedding_2d.dtype == torch.float32
    assert train_label_1d.shape == (train_size,)
    assert train_label_1d.dtype == torch.int64 
    assert val_embedding_2d.shape == (val_size, feat_dim)
    assert val_embedding_2d.dtype == torch.float32
    assert val_label_1d.shape == (val_size,)
    assert val_label_1d.dtype == torch.int64 
    assert test_embedding_2d.shape == (test_size, feat_dim)
    assert test_embedding_2d.dtype == torch.float32
    assert test_label_1d.shape == (test_size,)
    assert test_label_1d.dtype == torch.int64 
    train_embedding_2d = train_embedding_2d.detach() 
    val_embedding_2d = val_embedding_2d.detach()
    test_embedding_2d = test_embedding_2d.detach()

    device = train_embedding_2d.device 
    out_dim = int(torch.cat([train_label_1d, val_label_1d, test_label_1d]).max() + 1)
    
    model = MLP(
        in_dim = feat_dim, 
        hidden_dim = feat_dim,
        out_dim = out_dim,
        num_layers = num_layers,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    recorder = SupervisedRecorder(
        main_metric = main_metric,
        metrics = metrics,
        early_stopping_patience = early_stopping_patience,
        mute = True,
    )
    
    for epoch in tqdm(range(num_epochs), desc='run_linear_probe', disable=not use_tqdm):
        recorder.start_epoch()

        train_logit_2d = model(train_embedding_2d)
        
        train_loss = compute_cross_entropy_loss(logit_2d=train_logit_2d, label_1d=train_label_1d)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            val_logit_2d = model(val_embedding_2d)
            
        with torch.no_grad():
            test_logit_2d = model(test_embedding_2d)
            
        recorder.end_epoch(
            epoch = epoch,
            train_loss = train_loss,
            train_logit_2d = train_logit_2d.detach(),
            train_label_1d = train_label_1d,
            val_logit_2d = val_logit_2d.detach(),
            val_label_1d = val_label_1d,
            test_logit_2d = test_logit_2d.detach(),
            test_label_1d = test_label_1d,
        )

        if recorder.check_early_stopping():
            break
            
    linear_probe_result = recorder.summarize()
    print(f"\n{linear_probe_result = }\n")

    return linear_probe_result 

import torch 
from torch import Tensor 
import torchmetrics.functional.classification as F


def compute_binary_auc(
    input: Tensor,
    target: Tensor,
) -> float:
    if input.ndim == 2:
        assert input.shape[1] == 2 
        input = torch.softmax(input, dim=-1)[:, 1]

    batch_size, = input.shape 
    assert target.shape == (batch_size,) 
    input = input.detach()

    auc = F.binary_auroc(
        preds = input,
        target = target, 
    )

    return float(auc)


def compute_binary_ap(
    input: Tensor,
    target: Tensor,
) -> float:
    if input.ndim == 2:
        assert input.shape[1] == 2 
        input = torch.softmax(input, dim=-1)[:, 1]

    batch_size, = input.shape 
    assert target.shape == (batch_size,) 
    input = input.detach()

    ap = F.binary_average_precision(
        preds = input,
        target = target, 
    )

    return float(ap)


def compute_multiclass_acc(
    input: Tensor,
    target: Tensor,
) -> float:
    batch_size, num_classes = input.shape  
    assert target.shape == (batch_size,) 
    input = input.detach() 

    acc = F.multiclass_accuracy(
        preds = input,
        target = target,
    )

    return float(acc)


def compute_multiclass_micro_f1(
    input: Tensor,
    target: Tensor,
) -> float:
    batch_size, num_classes = input.shape  
    assert target.shape == (batch_size,) 
    input = input.detach() 

    f1 = F.multiclass_f1_score(
        preds = input,
        target = target,
        num_classes = num_classes,
        average = 'micro',
    )

    return float(f1)


def compute_multiclass_macro_f1(
    input: Tensor,
    target: Tensor,
) -> float:
    batch_size, num_classes = input.shape  
    assert target.shape == (batch_size,) 
    input = input.detach() 

    f1 = F.multiclass_f1_score(
        preds = input,
        target = target,
        num_classes = num_classes,
        average = 'macro',
    )

    return float(f1)


def compute_train_val_test_metrics(
    metric_list: list[str],
    train_input: Tensor,
    train_target: Tensor,
    val_input: Tensor,
    val_target: Tensor,
    test_input: Tensor,
    test_target: Tensor,
) -> dict[str, float]:
    metric_dict = dict() 

    for metric in metric_list:
        if metric == 'binary_auc':
            metric_func = compute_binary_auc
        elif metric == 'binary_ap':
            metric_func = compute_binary_ap
        elif metric == 'multiclass_acc':
            metric_func = compute_multiclass_acc
        elif metric == 'multiclass_micro_f1':
            metric_func = compute_multiclass_micro_f1
        elif metric == 'multiclass_macro_f1':
            metric_func = compute_multiclass_macro_f1
        else:
            raise ValueError 
        
        train_metric = metric_func(input=train_input, target=train_target)
        val_metric = metric_func(input=val_input, target=val_target)
        test_metric = metric_func(input=test_input, target=test_target)
        metric_dict[f'train_{metric}'] = train_metric
        metric_dict[f'val_{metric}'] = val_metric
        metric_dict[f'test_{metric}'] = test_metric
        
    return metric_dict 

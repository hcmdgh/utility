import torch 
from torch import Tensor 
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def compute_auc(
    logit_2d: Tensor,
    label_1d: Tensor,
) -> float:
    batch_size = len(logit_2d)
    assert logit_2d.dtype == torch.float32 and label_1d.dtype == torch.int64 
    assert logit_2d.shape == (batch_size, 2)
    assert label_1d.shape == (batch_size,) 
    assert torch.all(label_1d >= 0) and torch.all(label_1d <= 1)
    logit_2d = logit_2d.detach()

    torch.nan_to_num_(logit_2d, nan=0., posinf=0., neginf=0.)

    logit_2d = torch.softmax(logit_2d, dim=-1)
    logit_1d = logit_2d[:, 1]
    assert logit_1d.shape == (batch_size,)

    auc = roc_auc_score(
        y_true = label_1d.cpu().numpy(), 
        y_score = logit_1d.cpu().numpy(), 
    )

    return float(auc)


def compute_acc(
    logit_2d: Tensor,
    label_1d: Tensor,
) -> float:
    batch_size, num_classes = logit_2d.shape 
    assert logit_2d.dtype == torch.float32 and label_1d.dtype == torch.int64 
    assert label_1d.shape == (batch_size,) 
    logit_2d = logit_2d.detach() 

    torch.nan_to_num_(logit_2d, nan=0., posinf=0., neginf=0.)

    pred_1d = torch.argmax(logit_2d, dim=-1)
    assert pred_1d.shape == (batch_size,)

    acc = (pred_1d == label_1d).float().mean() 

    return float(acc)


def compute_ap(
    logit_2d: Tensor,
    label_1d: Tensor,
    error_return_negative_one: bool = False, 
) -> float:
    batch_size, num_classes = logit_2d.shape 
    logit_2d = logit_2d.detach()

    if num_classes > 2:
        if error_return_negative_one:
            return -1.
        else:
            raise ValueError 

    assert logit_2d.dtype == torch.float32 and label_1d.dtype == torch.int64 
    assert num_classes == 2 
    assert label_1d.shape == (batch_size,) 
    assert torch.all(label_1d >= 0) and torch.all(label_1d <= 1)

    torch.nan_to_num_(logit_2d, nan=0., posinf=0., neginf=0.)

    logit_2d = torch.softmax(logit_2d, dim=-1)
    logit_1d = logit_2d[:, 1]
    assert logit_1d.shape == (batch_size,)

    ap = average_precision_score(
        y_true = label_1d.cpu().numpy(), 
        y_score = logit_1d.cpu().numpy(), 
    )

    return float(ap)


def compute_micro_f1(
    logit_2d: Tensor,
    label_1d: Tensor,
) -> float:
    return compute_acc(
        logit_2d = logit_2d,
        label_1d = label_1d,
    )


def compute_macro_f1(
    logit_2d: Tensor,
    label_1d: Tensor,
) -> float:
    batch_size, num_classes = logit_2d.shape 
    assert logit_2d.dtype == torch.float32 and label_1d.dtype == torch.int64 
    assert label_1d.shape == (batch_size,) 
    logit_2d = logit_2d.detach()

    pred_1d = torch.argmax(logit_2d, dim=-1)
    assert pred_1d.shape == (batch_size,)

    macro_f1 = f1_score(
        y_true = label_1d.detach().cpu().numpy(),
        y_pred = pred_1d.detach().cpu().numpy(), 
        average = 'macro',
    )

    return float(macro_f1)

import torch 
from torch import Tensor 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


def run_kmeans(
    embedding_2d: Tensor,
    label_1d: Tensor, 
    init_cnt: int = 5,
) -> dict[str, float]:
    batch_size, embedding_dim = embedding_2d.shape  
    assert label_1d.shape == (batch_size,) 
    assert embedding_2d.dtype == torch.float32 and label_1d.dtype == torch.int64 
    embedding_np_2d = embedding_2d.detach().cpu().numpy()
    label_np_1d = label_1d.detach().cpu().numpy()

    num_classes = int(label_1d.max() + 1)

    kmeans = KMeans(
        n_clusters = num_classes, 
        init = 'k-means++', 
        random_state = 42, 
        n_init = init_cnt,
    )

    pred_1d = kmeans.fit_predict(embedding_np_2d)
    assert pred_1d.shape == (batch_size,)

    nmi = normalized_mutual_info_score(labels_pred=pred_1d, labels_true=label_np_1d)
    ari = adjusted_rand_score(labels_pred=pred_1d, labels_true=label_np_1d)

    return dict(
        nmi = float(nmi), 
        ari = float(ari),
    )


if __name__ == '__main__':
    x = torch.rand(10000, 256)
    y = torch.randint(low=0, high=10, size=(10000,))

    print(run_kmeans(
        embedding_2d = x,
        label_1d = y,
    ))

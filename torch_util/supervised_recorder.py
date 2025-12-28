import torch 
from torch import Tensor 
import time 
from typing import Any 

from .metric import compute_micro_f1, compute_macro_f1, compute_ap, compute_acc 


class SupervisedRecorder:
    def __init__(
        self, 
        main_metric: str,
        metrics: list[str],
        early_stopping_patience: int,
        mute: bool = False,
    ):
        self.main_metric = main_metric 
        self.metrics = metrics 
        self.early_stopping_patience = early_stopping_patience
        self.mute = mute 

        self.best_metric_dict = dict() 
        self.epoch_start_time = None 
        self.max_epoch = -1 

    def start_epoch(self):
        assert not self.epoch_start_time 
        self.epoch_start_time = time.perf_counter()

    @classmethod 
    def compute_metrics(
        cls,
        metrics: list[str],
        train_logit_2d: Tensor,
        train_label_1d: Tensor,
        val_logit_2d: Tensor,
        val_label_1d: Tensor,
        test_logit_2d: Tensor,
        test_label_1d: Tensor,
    ) -> dict[str, float]:
        metric_dict = dict() 

        for prefix, logit_2d, label_1d in [
            ('train', train_logit_2d, train_label_1d),
            ('val', val_logit_2d, val_label_1d),
            ('test', test_logit_2d, test_label_1d),
        ]:
            for metric in metrics:
                if metric == 'micro_f1':
                    value = compute_micro_f1(logit_2d=logit_2d, label_1d=label_1d)
                elif metric == 'macro_f1':
                    value = compute_macro_f1(logit_2d=logit_2d, label_1d=label_1d)  
                elif metric == 'ap':
                    value = compute_ap(logit_2d=logit_2d, label_1d=label_1d) 
                elif metric == 'acc':
                    value = compute_acc(logit_2d=logit_2d, label_1d=label_1d) 
                else:
                    raise ValueError 
                
                metric_dict[f'{prefix}_{metric}'] = value

        return metric_dict
    
    def end_epoch(
        self,
        epoch: int,
        train_loss: Any,
        train_logit_2d: Tensor,
        train_label_1d: Tensor,
        val_logit_2d: Tensor,
        val_label_1d: Tensor,
        test_logit_2d: Tensor,
        test_label_1d: Tensor,
    ):
        if isinstance(train_loss, Tensor):
            train_loss = train_loss.item() 
        train_loss = float(train_loss)

        metric_dict = dict(
            epoch = epoch,
            train_loss = train_loss,
        )

        metric_dict.update(
            self.compute_metrics(
                metrics = self.metrics,
                train_logit_2d = train_logit_2d,
                train_label_1d = train_label_1d,
                val_logit_2d = val_logit_2d,
                val_label_1d = val_label_1d,
                test_logit_2d = test_logit_2d,
                test_label_1d = test_label_1d,
            )
        ) 

        assert epoch > self.max_epoch 
        self.max_epoch = epoch 

        assert self.epoch_start_time 
        metric_dict['epoch_duration'] = time.perf_counter() - self.epoch_start_time 
        self.epoch_start_time = None 

        if not self.best_metric_dict or metric_dict[f"val_{self.main_metric}"] > self.best_metric_dict[f"val_{self.main_metric}"]:
            self.best_metric_dict = metric_dict

        if not self.mute:
            print(metric_dict)

    def check_early_stopping(self) -> bool:
        if self.early_stopping_patience <= 0:
            return False 
        
        assert self.best_metric_dict
        best_epoch = self.best_metric_dict['epoch'] 
        assert self.max_epoch >= best_epoch
        return self.max_epoch - best_epoch > self.early_stopping_patience  

    def summarize(self) -> dict:
        assert self.best_metric_dict
        self.best_metric_dict['max_epoch'] = self.max_epoch

        if not self.mute:
            print() 
            print("Best Epoch:")
            
            for key, value in self.best_metric_dict.items():
                print(f"  {key}: {value}")

            print() 

        return self.best_metric_dict

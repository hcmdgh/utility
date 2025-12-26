import torch 
from torch import Tensor 
import time 
from typing import Any 

from .linear_probe import run_linear_probe


class UnsupervisedRecorder:
    def __init__(
        self, 
        main_metric: str,
        metrics: list[str],
        linear_probe_lr: float,
        linear_probe_num_epochs: int,
        early_stopping_patience: int,
        mute: bool = False,
    ):
        self.main_metric = main_metric 
        self.metrics = metrics 
        self.linear_probe_lr = linear_probe_lr
        self.linear_probe_num_epochs = linear_probe_num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.mute = mute 

        self.best_metric_dict = dict() 
        self.epoch_start_time = None 
        self.max_epoch = -1 

    def start_epoch(self):
        assert not self.epoch_start_time 
        self.epoch_start_time = time.perf_counter()

    def end_epoch(
        self,
        epoch: int,
        train_loss: Any,
        train_embedding_2d: Tensor,
        train_label_1d: Tensor,
        val_embedding_2d: Tensor,
        val_label_1d: Tensor,
        test_embedding_2d: Tensor,
        test_label_1d: Tensor,
    ):
        if isinstance(train_loss, Tensor):
            train_loss = train_loss.item()
        train_loss = float(train_loss)

        metric_dict = dict(
            epoch = epoch,
            train_loss = train_loss,
        )

        linear_probe_result = run_linear_probe(
            train_embedding_2d = train_embedding_2d,
            train_label_1d = train_label_1d,
            val_embedding_2d = val_embedding_2d,
            val_label_1d = val_label_1d,
            test_embedding_2d = test_embedding_2d,
            test_label_1d = test_label_1d,
            main_metric = self.main_metric,
            metrics = self.metrics,
            lr = self.linear_probe_lr,
            num_epochs = self.linear_probe_num_epochs,
        )
        linear_probe_result = { 
            key: value 
            for key, value in linear_probe_result.items()
            if key.startswith('train_') or key.startswith('val_') or key.startswith('test_')
        }
        metric_dict.update(linear_probe_result)

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

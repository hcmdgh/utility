import torch 
from torch import Tensor 
import time 
from typing import Any 


class SingleMetricRecorder:
    def __init__(
        self, 
        metric_name: str,
        early_stopping_patience: int,
    ):
        self.metric_name = metric_name
        self.early_stopping_patience = early_stopping_patience

        self.best_val_metric = -1. 
        self.best_epoch = -1
        self.best_epoch_data = None  
        self.final_train_metric = -1. 
        self.final_test_metric = -1.
        self.epoch_start_time = None 
        self.max_epoch = -1 

    def start_epoch(self):
        assert not self.epoch_start_time 
        self.epoch_start_time = time.perf_counter()
    
    def end_epoch(
        self,
        epoch: int,
        train_loss: Any,  
        train_metric: Any,
        val_metric: Any,
        test_metric: Any,
        data: Any = None,
    ):
        if isinstance(train_loss, Tensor):
            train_loss = train_loss.item() 
            
        train_loss = float(train_loss)
        train_metric = float(train_metric)
        val_metric = float(val_metric)
        test_metric = float(test_metric)

        assert self.epoch_start_time 
        epoch_duration = time.perf_counter() - self.epoch_start_time 
        self.epoch_start_time = None 

        assert epoch > self.max_epoch 
        self.max_epoch = epoch 

        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric 
            self.best_epoch = epoch 
            self.best_epoch_data = data 
            self.final_train_metric = train_metric 
            self.final_test_metric = test_metric

        print(f"[epoch: {epoch}] train_loss: {train_loss:.5f}, train_{self.metric_name}: {train_metric:.2%}, val_{self.metric_name}: {val_metric:.2%}, test_{self.metric_name}: {test_metric:.2%}, duration: {epoch_duration:.3f}s")

    def check_early_stopping(self) -> bool:
        if self.early_stopping_patience <= 0:
            return False 

        assert self.max_epoch >= self.best_epoch 
        return self.max_epoch - self.best_epoch > self.early_stopping_patience  

    def summarize(self) -> dict:
        print() 
        print("[Summary]")
        print(f"  best_epoch: {self.best_epoch} / {self.max_epoch}")
        print(f"  final_train_{self.metric_name}: {self.final_train_metric:.2%}")
        print(f"  best_val_{self.metric_name}: {self.best_val_metric:.2%}")
        print(f"  final_test_{self.metric_name}: {self.final_test_metric:.2%}")
        print() 

        return dict(
            best_epoch = self.best_epoch,
            max_epoch = self.max_epoch,
            final_train_metric = self.final_train_metric,
            best_val_metric = self.best_val_metric,
            final_test_metric = self.final_test_metric, 
            best_epoch_data = self.best_epoch_data,
        )

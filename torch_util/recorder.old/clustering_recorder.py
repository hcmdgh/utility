import torch 
from torch import Tensor 
import time 
import copy 
from typing import Any 

from .kmeans import run_kmeans 


class ClusteringRecorder:
    def __init__(
        self, 
        early_stopping_patience: int,
        mute: bool = False,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.mute = mute 

        self.best_epoch = -1 
        self.best_nmi = -1. 
        self.best_ari = -1. 
        self.best_data = None 

        self.epoch_start_time = None 
        self.max_epoch = -1 

    def start_epoch(self):
        assert not self.epoch_start_time 
        self.epoch_start_time = time.perf_counter()

    def end_epoch(
        self,
        epoch: int,
        train_loss: Any,
        embedding_2d: Tensor,
        label_1d: Tensor,
        data: Any = None, 
    ):
        if isinstance(train_loss, Tensor):
            train_loss = train_loss.item()
        train_loss = float(train_loss)

        kmeans_result = run_kmeans(
            embedding_2d = embedding_2d,
            label_1d = label_1d,
        )
        nmi = kmeans_result['nmi']
        ari = kmeans_result['ari']

        assert epoch > self.max_epoch 
        self.max_epoch = epoch 

        assert self.epoch_start_time 
        epoch_duration = time.perf_counter() - self.epoch_start_time 
        self.epoch_start_time = None 

        if ari > self.best_ari:
            self.best_epoch = epoch 
            self.best_nmi = nmi 
            self.best_ari = ari
            self.best_data = copy.deepcopy(data)  

        if not self.mute:
            epoch_dict = dict(
                epoch = epoch,
                train_loss = train_loss,
                nmi = nmi,
                ari = ari,
                duration = epoch_duration,
            )
            print(epoch_dict)

    def check_early_stopping(self) -> bool:
        if self.early_stopping_patience <= 0:
            return False 
        
        assert self.best_epoch > -1 
        assert self.max_epoch >= self.best_epoch

        return self.max_epoch - self.best_epoch > self.early_stopping_patience  

    def summarize(self) -> dict:
        summary_dict = dict(
            max_epoch = self.max_epoch,
            best_epoch = self.best_epoch,
            best_nmi = self.best_nmi,
            best_ari = self.best_ari,
            best_data = self.best_data,
        )

        if not self.mute:
            print() 
            print("[Summary]")
            
            for key, value in summary_dict.items():
                print(f"  {key}: {value}")

            print() 

        return summary_dict

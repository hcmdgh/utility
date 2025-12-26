import torch 
from torch import Tensor 
import time 
from typing import Any 

from ..metric import compute_macro_f1, compute_micro_f1, compute_ap  


class SupervisedRecorder:
    def __init__(
        self, 
        metrics: str,
        early_stopping_patience: int,
        mute: bool = False,
    ):
        self.metrics = metrics 
        self.early_stopping_patience = early_stopping_patience
        self.mute = mute 

        self.best_state_dict = dict() 
        self.epoch_start_time = None 
        self.max_epoch = -1 

    def start_epoch(self):
        assert not self.epoch_start_time 
        self.epoch_start_time = time.perf_counter()
    
    def end_epoch(
        self,
        epoch: int,
        train_loss: Tensor,
        train_logit_2d: Tensor,
        train_label_1d: Tensor,
        val_logit_2d: Tensor,
        val_label_1d: Tensor,
        test_logit_2d: Tensor,
        test_label_1d: Tensor,
    ):
        if self.metrics == 'micro_macro_f1_ap':
            epoch_state_dict = dict(
                epoch = epoch,
                train_loss = train_loss.item(),
                train_micro_f1 = compute_micro_f1(logit_2d=train_logit_2d, label_1d=train_label_1d),
                train_macro_f1 = compute_macro_f1(logit_2d=train_logit_2d, label_1d=train_label_1d),
                train_ap = compute_ap(logit_2d=train_logit_2d, label_1d=train_label_1d),
                val_micro_f1 = compute_micro_f1(logit_2d=val_logit_2d, label_1d=val_label_1d),
                val_macro_f1 = compute_macro_f1(logit_2d=val_logit_2d, label_1d=val_label_1d),
                val_ap = compute_ap(logit_2d=val_logit_2d, label_1d=val_label_1d),
                test_micro_f1 = compute_micro_f1(logit_2d=test_logit_2d, label_1d=test_label_1d),
                test_macro_f1 = compute_macro_f1(logit_2d=test_logit_2d, label_1d=test_label_1d),
                test_ap = compute_ap(logit_2d=test_logit_2d, label_1d=test_label_1d),
            )

            if epoch_state_dict['val_ap'] > self.best_state_dict.get('val_ap', -1):
                self.best_state_dict = epoch_state_dict
        elif self.metrics == 'micro_macro_f1':
            epoch_state_dict = dict(
                epoch = epoch,
                train_loss = train_loss.item(),
                train_micro_f1 = compute_micro_f1(logit_2d=train_logit_2d, label_1d=train_label_1d),
                train_macro_f1 = compute_macro_f1(logit_2d=train_logit_2d, label_1d=train_label_1d),
                val_micro_f1 = compute_micro_f1(logit_2d=val_logit_2d, label_1d=val_label_1d),
                val_macro_f1 = compute_macro_f1(logit_2d=val_logit_2d, label_1d=val_label_1d),
                test_micro_f1 = compute_micro_f1(logit_2d=test_logit_2d, label_1d=test_label_1d),
                test_macro_f1 = compute_macro_f1(logit_2d=test_logit_2d, label_1d=test_label_1d),
            )

            if epoch_state_dict['val_micro_f1'] > self.best_state_dict.get('val_micro_f1', -1):
                self.best_state_dict = epoch_state_dict
        else:
            raise ValueError 

        assert self.epoch_start_time 
        epoch_duration = time.perf_counter() - self.epoch_start_time 
        self.epoch_start_time = None 

        assert epoch > self.max_epoch 
        self.max_epoch = epoch 

        if not self.mute:
            epoch_state_dict = epoch_state_dict.copy() 
            epoch_state_dict['epoch_duration'] = epoch_duration
            print(epoch_state_dict)

    def check_early_stopping(self) -> bool:
        if self.early_stopping_patience <= 0:
            return False 

        best_epoch = self.best_state_dict['epoch'] 
        assert self.max_epoch >= best_epoch
        return self.max_epoch - best_epoch > self.early_stopping_patience  

    def summarize(self) -> dict:
        self.best_state_dict['max_epoch'] = self.max_epoch

        if not self.mute:
            print() 
            print("Best Epoch:")
            print(self.best_state_dict)
            print() 

        return self.best_state_dict

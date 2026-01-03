import torch 
from torch import Tensor 
import time 
from typing import Any 


class MultipleMetricRecorder:
    def __init__(
        self, 
        early_stopping_patience: int,
        mute: bool = False,
    ):
        self.early_stopping_patience = early_stopping_patience
        self.mute = mute 

        self.best_epoch = -1 
        self.best_train_micro_f1 = -1.  
        self.best_train_macro_f1 = -1.  
        self.best_train_ap = -1.
        self.best_val_micro_f1 = -1.  
        self.best_val_macro_f1 = -1.  
        self.best_val_ap = -1.
        self.best_test_micro_f1 = -1.
        self.best_test_macro_f1 = -1.  
        self.best_test_ap = -1.  
        self.epoch_start_time = None 
        self.max_epoch = -1 

    def start_epoch(self):
        assert not self.epoch_start_time 
        self.epoch_start_time = time.perf_counter()
    
    def end_epoch(
        self,
        epoch: int,
        train_loss: Any,  
        train_micro_f1: float = -1.,
        train_macro_f1: float = -1.,
        train_ap: float = -1.,
        val_micro_f1: float = -1.,
        val_macro_f1: float = -1.,
        val_ap: float = -1.,
        test_micro_f1: float = -1.,
        test_macro_f1: float = -1.,
        test_ap: float = -1.,
    ):
        if isinstance(train_loss, Tensor):
            train_loss = train_loss.item() 
        train_loss = float(train_loss)

        assert self.epoch_start_time 
        epoch_duration = time.perf_counter() - self.epoch_start_time 
        self.epoch_start_time = None 

        assert epoch > self.max_epoch 
        self.max_epoch = epoch 

        if val_ap > -1: 
            update = val_ap > self.best_val_ap 
        elif val_micro_f1 > -1: 
            update = val_micro_f1 > self.best_val_micro_f1
        else:
            raise ValueError 

        if update:
            self.best_epoch = epoch 
            self.best_train_micro_f1 = train_micro_f1 
            self.best_train_macro_f1 = train_macro_f1 
            self.best_train_ap = train_ap 
            self.best_val_micro_f1 = val_micro_f1 
            self.best_val_macro_f1 = val_macro_f1
            self.best_val_ap = val_ap
            self.best_test_micro_f1 = test_micro_f1 
            self.best_test_macro_f1 = test_macro_f1 
            self.best_test_ap = test_ap  

        msg = f"[epoch: {epoch}] train_loss: {train_loss:.5f}" 

        if train_micro_f1 > -1:
            msg += f", train_micro_f1: {train_micro_f1:.2%}"
        if train_macro_f1 > -1:
            msg += f", train_macro_f1: {train_macro_f1:.2%}"
        if train_ap > -1:
            msg += f", train_ap: {train_ap:.2%}"
        if val_micro_f1 > -1:
            msg += f", val_micro_f1: {val_micro_f1:.2%}"
        if val_macro_f1 > -1:
            msg += f", val_macro_f1: {val_macro_f1:.2%}"
        if val_ap > -1:
            msg += f", val_ap: {val_ap:.2%}"
        if test_micro_f1 > -1:
            msg += f", test_micro_f1: {test_micro_f1:.2%}"
        if test_macro_f1 > -1:
            msg += f", test_macro_f1: {test_macro_f1:.2%}"
        if test_ap > -1:
            msg += f", test_ap: {test_ap:.2%}"

        msg += f", duration: {epoch_duration:.3f}s"

        if not self.mute:
            print(msg)

    def check_early_stopping(self) -> bool:
        if self.early_stopping_patience <= 0:
            return False 

        assert self.max_epoch >= self.best_epoch 
        return self.max_epoch - self.best_epoch > self.early_stopping_patience  

    def summarize(self) -> dict:
        if not self.mute:
            print() 
            print("[Summary]")
            print(f"  best_epoch: {self.best_epoch} / {self.max_epoch}")

            if self.best_train_micro_f1 > -1:
                print(f"  best_train_micro_f1: {self.best_train_micro_f1:.2%}")
            if self.best_train_macro_f1 > -1:
                print(f"  best_train_macro_f1: {self.best_train_macro_f1:.2%}")
            if self.best_train_ap > -1:
                print(f"  best_train_ap: {self.best_train_ap:.2%}")
            if self.best_val_micro_f1 > -1:
                print(f"  best_val_micro_f1: {self.best_val_micro_f1:.2%}")
            if self.best_val_macro_f1 > -1:
                print(f"  best_val_macro_f1: {self.best_val_macro_f1:.2%}")
            if self.best_val_ap > -1:
                print(f"  best_val_ap: {self.best_val_ap:.2%}")
            if self.best_test_micro_f1 > -1:
                print(f"  best_test_micro_f1: {self.best_test_micro_f1:.2%}")
            if self.best_test_macro_f1 > -1:
                print(f"  best_test_macro_f1: {self.best_test_macro_f1:.2%}")
            if self.best_test_ap > -1:
                print(f"  best_test_ap: {self.best_test_ap:.2%}")

            print() 

        return dict(
            best_epoch = self.best_epoch,
            max_epoch = self.max_epoch,
            best_train_micro_f1 = self.best_train_micro_f1,
            best_train_macro_f1 = self.best_train_macro_f1,
            best_train_ap = self.best_train_ap,
            best_val_micro_f1 = self.best_val_micro_f1,
            best_val_macro_f1 = self.best_val_macro_f1,
            best_val_ap = self.best_val_ap,
            best_test_micro_f1 = self.best_test_micro_f1,
            best_test_macro_f1 = self.best_test_macro_f1,
            best_test_ap = self.best_test_ap,
        )

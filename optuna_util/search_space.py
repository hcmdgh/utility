import optuna 
import json 
from dataclasses import dataclass
from typing import Optional, Any 


class SearchSpace:
    def suggest(
        self,
        trial: optuna.Trial,
        name: str,
    ) -> Any:
        raise NotImplementedError 


@dataclass 
class IntSearchSpace(SearchSpace):
    low: int 
    high: int 
    step: int = 1   

    def suggest(
        self,
        trial: optuna.Trial,
        name: str,
    ) -> int:
        return trial.suggest_int(name=name, low=self.low, high=self.high, step=self.step) 


@dataclass
class FloatSearchSpace(SearchSpace):
    low: float 
    high: float 
    step: Optional[float] = None  
    log: bool = False 

    def suggest(
        self,
        trial: optuna.Trial,
        name: str,
    ) -> float:
        return trial.suggest_float(name=name, low=self.low, high=self.high, step=self.step, log=self.log)


@dataclass
class CategoricalSearchSpace(SearchSpace):
    choices: list 

    def suggest(
        self,
        trial: optuna.Trial,
        name: str,
    ) -> Any:
        if isinstance(self.choices[0], (list, tuple, dict)):
            choices = [
                json.dumps(item, ensure_ascii=False)  
                for item in self.choices
            ]
        else:
            choices = self.choices

        return trial.suggest_categorical(name=name, choices=choices)


@dataclass 
class BoolSearchSpace(CategoricalSearchSpace):
    def __init__(self):
        super().__init__(choices=[True, False])

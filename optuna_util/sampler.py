import optuna 
from typing import Optional 


def create_sampler(
    type: str,
    search_space: Optional[dict] = None,
) -> optuna.samplers.BaseSampler:
    if type == 'random':
        return optuna.samplers.RandomSampler()
    elif type == 'tpe':
        return optuna.samplers.TPESampler()
    elif type == 'grid':
        assert search_space
        return optuna.samplers.GridSampler(search_space=search_space)
    else:
        raise ValueError 

import optuna 
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import os
import traceback 
import json 
from queue import Queue 
from functools import partial
from typing import Callable, Optional

from .search_space import SearchSpace, CategoricalSearchSpace, IntSearchSpace
from .sampler import create_sampler 
from .pruner import RepeatParamPruner
from .study import count_completed_trials, optimize_callback_func


def _objective_func(
    trial: optuna.Trial,
    trial_func: Callable,
    search_space_dict: dict,
    device_queue: Queue[int],
) -> float:
    device_id = device_queue.get()

    param_dict = dict() 

    for param_name, search_space in search_space_dict.items():
        if isinstance(search_space, SearchSpace):
            param_dict[param_name] = search_space.suggest(trial=trial, name=param_name)
        else:
            param_dict[param_name] = search_space 

    try:
        objective_value, other_result_dict = trial_func(
            trial = trial, 
            device_id = device_id,
            params = param_dict,
        )
    except optuna.TrialPruned:
        raise 
    except Exception:
        traceback.print_exc()
        return float('nan')
    else:
        for key, value in other_result_dict.items():
            trial.set_user_attr(key=key, value=value)

        return objective_value
    finally:
        device_queue.put(device_id)


def start_optuna_search(
    study_name: str,
    device_ids: list[int],
    sampler: str,
    trial_func: Callable,
    search_space_dict: dict,
    storage_uri: str,
    output_dir: str, 
    optimize_direction: str = 'maximize',
    num_trials: Optional[int] = None,
):
    if sampler == 'grid':
        grid_search_space = dict() 

        for k, v in search_space_dict.items():
            if isinstance(v, SearchSpace):
                if isinstance(v, CategoricalSearchSpace):
                    grid_search_space[k] = v.choices
                elif isinstance(v, IntSearchSpace):
                    assert v.low <= v.high and v.step > 0
                    grid_search_space[k] = list(range(v.low, v.high + 1, v.step))
                else:
                    raise TypeError 
    else:
        grid_search_space = dict() 

    sampler_obj = create_sampler(sampler, search_space=grid_search_space)

    pruner = RepeatParamPruner(sampler=sampler)

    device_queue = Queue() 

    for device_id in device_ids:
        device_queue.put(device_id)

    os.makedirs(output_dir, exist_ok=True)
    
    study = optuna.create_study(
        study_name = study_name,
        sampler = sampler_obj,
        pruner = pruner,
        # storage = JournalStorage(JournalFileBackend(os.path.join(output_dir, 'storage.log'))),
        storage = storage_uri,
        direction = optimize_direction,
        load_if_exists = True,
    )

    if isinstance(sampler_obj, optuna.samplers.GridSampler):
        if sampler_obj.is_exhausted(study=study):
            return 
        
    study.optimize(
        func = partial(
            _objective_func,
            trial_func = trial_func,
            search_space_dict = search_space_dict,
            device_queue = device_queue,
        ),
        n_jobs = len(device_ids),
        callbacks = [
            partial(
                optimize_callback_func,
                num_trials = num_trials,
            )
        ],
        gc_after_trial = True,
    )

    df = study.trials_dataframe()
    df = df.sort_values("value", ascending=(optimize_direction != 'maximize'))
    df.to_csv(os.path.join(output_dir, 'search_result.csv'), index=False)

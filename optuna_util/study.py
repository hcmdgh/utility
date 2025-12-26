import optuna 
from typing import Optional 


def count_completed_trials(study: optuna.Study) -> int:
    completed_cnt = 0
    
    for trial in study.get_trials(deepcopy=False):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            completed_cnt += 1
            
    return completed_cnt


def optimize_callback_func(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    num_trials: Optional[int],
):
    if num_trials:
        if count_completed_trials(study=study) >= num_trials:
            study.stop()

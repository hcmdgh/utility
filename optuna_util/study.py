import optuna 
from typing import Optional 


def count_completed_trials(study: optuna.Study) -> int:
    completed_cnt = 0
    
    for trial in study.get_trials(deepcopy=False):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            completed_cnt += 1
            
    return completed_cnt


def check_study_completed(
    study: optuna.Study,
    trial: optuna.trial.FrozenTrial,
    num_trials: Optional[int],
):
    if num_trials:
        if count_completed_trials(study=study) >= num_trials:
            study.stop()


def is_trial_duplicated(
    trial: optuna.Trial,
) -> bool:
    study = trial.study
    trial_list = study.get_trials(deepcopy=False)
    existing_trial_list = [t for t in trial_list if t.number < trial.number]
    existing_trial_params_list = [t.params for t in existing_trial_list]

    if trial.params in existing_trial_params_list:
        return True
    else:
        return False

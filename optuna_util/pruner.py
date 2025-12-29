import optuna 


class RepeatParamPruner(optuna.pruners.BasePruner):
    def __init__(
        self,
        sampler: str,
    ):
        super().__init__() 

        self.enable = sampler != 'grid'
        # self.enable = True 

    def prune(
        self, 
        study: optuna.Study, 
        trial: optuna.trial.FrozenTrial,
    ) -> bool:
        if not self.enable:
            return False

        trial_list = study.get_trials(deepcopy=False)
        existing_trial_list = [t for t in trial_list if t.number < trial.number]
        existing_trial_params_list = [t.params for t in existing_trial_list]

        if trial.params in existing_trial_params_list:
            return True
        else:
            return False

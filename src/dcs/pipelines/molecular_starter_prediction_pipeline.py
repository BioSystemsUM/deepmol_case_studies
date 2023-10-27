from typing import List, Tuple

import optuna
from deepmol.datasets import SmilesDataset
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.pipeline_optimization._utils import preset_all_models, preset_sklearn_models, preset_keras_models

from dcs.alkaloid_data.alkaloid_data_objective import MolecularStartersObjective


def run_molecular_starter_prediction_pipeline(cv_data: List[Tuple[SmilesDataset, SmilesDataset]],
                                              optimizer: str = 'tpe', seed: int = 1, timeout: int = 60*3,
                                              n_trials: int = 100, save_top_n: int = 3):

    if optimizer == 'nsga2':
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    elif optimizer == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f'Invalid optimizer: {optimizer}. It must be one of "nsga2" or "tpe"')

    pipeline_name = f'molecular_starter_prediction_pipeline_attempt_keras'

    pipeline = PipelineOptimization(sampler=sampler,
                                    study_name=pipeline_name,
                                    direction='maximize',
                                    storage='sqlite:///pipeline_test.db')

    def objective_steps(trial: optuna.Trial, data):
        return preset_all_models(trial, data)

    def objective_steps_sklearn(trial: optuna.Trial, data):
        return preset_sklearn_models(trial, data)

    def objective_steps_keras(trial: optuna.Trial, data):
        return preset_keras_models(trial, data)

    pipeline.optimize(cv_data=cv_data, objective_steps=objective_steps_keras,
                      n_trials=n_trials, save_top_n=save_top_n, objective=MolecularStartersObjective,
                      trial_timeout=timeout)
    return pipeline.best_pipeline

import time
from typing import List, Tuple

import optuna
from deepmol.datasets import SmilesDataset
from deepmol.pipeline_optimization import PipelineOptimization
from deepmol.pipeline_optimization._utils import preset_all_models

from dcs.alkaloid_data.alkaloid_data_objective import MolecularStartersObjective


def run_molecular_starter_prediction_pipeline(cv_data: List[Tuple[SmilesDataset, SmilesDataset]],
                                              optimizer: str = 'tpe', seed: int = 1):

    if optimizer == 'nsga2':
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    elif optimizer == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f'Invalid optimizer: {optimizer}. It must be one of "nsga2" or "tpe"')

    pipeline_name = f'molecular_starter_prediction_pipeline_{time.strftime("%Y_%m_%d-%H_%M_%S")}'

    pipeline = PipelineOptimization(sampler=sampler,
                                    study_name=pipeline_name,
                                    direction='maximize')

    def objective_steps(trial: optuna.Trial, data):
        return preset_all_models(trial, data)

    pipeline.optimize(cv_data=cv_data, objective_steps=objective_steps,
                      n_trials=100, save_top_n=3, objective=MolecularStartersObjective, trial_timeout=60 * 3)
    return pipeline.best_pipeline

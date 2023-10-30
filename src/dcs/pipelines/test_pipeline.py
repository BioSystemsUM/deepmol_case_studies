import time

from deepmol.datasets import Dataset
from deepmol.loggers import Logger
from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
import optuna
from deepmol.pipeline_optimization._utils import preset_all_models
from sklearn.metrics import roc_auc_score

from dcs.tdc.tdc_objective import TDCObjective


def test_pipeline(pipeline_name: str = None, group=None, tdc_dataset_name: str = None, data_sample: Dataset = None,
                  seed: int = 1, optimizer: str = 'tpe', metric=roc_auc_score, direction: str = 'maximize',
                  n_trials: int = 100, save_top_n: int = 1, trial_timeout: int = 60*3):
    Logger().disable()
    if optimizer == 'nsga2':
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    elif optimizer == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f'Invalid optimizer: {optimizer}. It must be one of "nsga2" or "tpe"')
    pipeline_name = pipeline_name if pipeline_name is not None else f'pipeline_{time.strftime("%Y_%m_%d-%H_%M_%S")}'
    metric = Metric(metric)
    pipeline = PipelineOptimization(sampler=sampler,
                                    study_name=pipeline_name,
                                    direction=direction)

    def objective_steps(trial: optuna.Trial, data):
        return preset_all_models(trial, data)

    pipeline.optimize(objective_steps=objective_steps, n_trials=n_trials, save_top_n=save_top_n,
                      objective=TDCObjective, trial_timeout=trial_timeout, metric=metric, group=group,
                      tdc_dataset_name=tdc_dataset_name, data=data_sample)
    print(pipeline.trials_dataframe())
    print(f"Best trial: {pipeline.best_trial}")
    print(f"Best score: {pipeline.best_value}")
    print(f"Best params: {pipeline.best_params}")
    # save trials_dataframe to csv
    pipeline.trials_dataframe().to_csv(f'{pipeline_name}_trials.csv')
    return pipeline.best_pipeline

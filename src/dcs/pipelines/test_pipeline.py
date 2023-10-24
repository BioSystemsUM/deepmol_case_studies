import time

from deepmol.datasets import SmilesDataset, Dataset
from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
import optuna
from deepmol.pipeline_optimization._utils import preset_all_models
from sklearn.metrics import roc_auc_score

from dcs.tdc.tdc_objective import TDCObjective


def test_pipeline(pipeline_name: str = None, group=None, tdc_dataset_name: str = None, data_sample: Dataset = None,
                  seed: int = 1, optimizer: str = 'tpe'):
    if optimizer == 'nsga2':
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    elif optimizer == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f'Invalid optimizer: {optimizer}. It must be one of "nsga2" or "tpe"')
    pipeline_name = pipeline_name or f'pipeline_{time.strftime("%Y_%m_%d-%H_%M_%S")}'
    metric = Metric(roc_auc_score)
    pipeline = PipelineOptimization(sampler=sampler,
                                    study_name=pipeline_name,
                                    direction='maximize')

    def objective_steps(trial: optuna.Trial, data):
        return preset_all_models(trial, data)

    pipeline.optimize(train_dataset=None, test_dataset=None, objective_steps=objective_steps, metric=metric,
                      n_trials=10, save_top_n=3, objective=TDCObjective, trial_timeout=60*60, group=group,
                      tdc_dataset_name=tdc_dataset_name, data=data_sample)
    return pipeline.best_pipeline

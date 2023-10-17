import time

from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
import optuna
from sklearn.metrics import roc_auc_score


def test_pipeline(train, valid, pipeline_name: str = None, seed: int = 1, optimizer: str = 'tpe'):
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
    pipeline.optimize(train, valid, objective_steps='all', metric=metric, n_trials=10, save_top_n=3, data=train)
    return pipeline.best_pipeline

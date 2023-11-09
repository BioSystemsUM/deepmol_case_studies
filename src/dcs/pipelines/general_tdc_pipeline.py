import os
import time

from deepmol.datasets import Dataset, SmilesDataset
from deepmol.loggers import Logger
from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
import optuna
from deepmol.pipeline_optimization._utils import preset_all_models
from sklearn.metrics import roc_auc_score

from dcs.objectives import TDCObjective


def general_tdc_pipeline(pipeline_name: str = None, group=None, tdc_dataset_name: str = None,
                         data_sample: Dataset = None, seed: int = 1, optimizer: str = 'tpe', storage: str = None,
                         metric: callable = roc_auc_score, direction: str = 'maximize', n_trials: int = 100,
                         save_top_n: int = 1, trial_timeout: int = 60 * 3):
    # Logger().disable()
    if optimizer == 'nsga2':
        sampler = optuna.samplers.NSGAIISampler(seed=seed)
    elif optimizer == 'tpe':
        sampler = optuna.samplers.TPESampler(seed=seed)
    else:
        raise ValueError(f'Invalid optimizer: {optimizer}. It must be one of "nsga2" or "tpe"')
    pipeline_name = pipeline_name if pipeline_name is not None else f'pipeline_{time.strftime("%Y_%m_%d-%H_%M_%S")}'
    # create the directory pipeline_name
    os.makedirs(pipeline_name, exist_ok=True)
    storage = storage if storage is not None else f'sqlite:///{pipeline_name}.db'
    metric = Metric(metric)
    pipeline = PipelineOptimization(storage=storage,
                                    sampler=sampler,
                                    study_name=pipeline_name,
                                    direction=direction)

    def objective_steps(trial: optuna.Trial, data):
        return preset_all_models(trial, data)

    # get splits
    benchmark = group.get(tdc_dataset_name)
    name = benchmark['name']
    train_sets, valid_sets, test_sets = [], [], []
    for seed in [1, 2, 3, 4, 5]:
        train_val, test = benchmark['train_val'], benchmark['test']
        train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)
        train_sets.append(SmilesDataset(smiles=train['Drug'].values, ids=train['Drug_ID'].values,
                                        y=train['Y'].values))
        valid_sets.append(SmilesDataset(smiles=valid['Drug'].values, ids=valid['Drug_ID'].values,
                                        y=valid['Y'].values))
        test_sets.append(SmilesDataset(smiles=test['Drug'].values, ids=test['Drug_ID'].values,
                                       y=test['Y'].values))

    pipeline.optimize(objective_steps=objective_steps, n_trials=n_trials, save_top_n=save_top_n,
                      objective=TDCObjective, trial_timeout=trial_timeout, metric=metric, group=group,
                      tdc_dataset_name=tdc_dataset_name, data=data_sample, splits=[train_sets, valid_sets, test_sets])
    print(pipeline.trials_dataframe())
    print(f"Best trial: {pipeline.best_trial}")
    print(f"Best score: {pipeline.best_value}")
    print(f"Best params: {pipeline.best_params}")
    # save trials_dataframe to csv
    pipeline.trials_dataframe().to_csv(f'{pipeline_name}_trials.csv')
    return pipeline.best_pipeline

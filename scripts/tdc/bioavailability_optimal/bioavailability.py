import time

from deepmol.datasets import SmilesDataset
from sklearn.metrics import roc_auc_score

from deepmol_models.pipelines import general_tdc_pipeline
from deepmol_models.utils import get_benchmark_group

from deepmol.pipeline_optimization._feature_selector_objectives import _get_feature_selector, percentil_fs
from deepmol.pipeline_optimization._featurizer_objectives import _get_featurizer
from deepmol.pipeline_optimization._scaler_objectives import _get_scaler
from deepmol.pipeline_optimization._sklearn_model_objectives import hist_gradient_boosting_classifier_step
from deepmol.pipeline_optimization._standardizer_objectives import _get_standardizer
from deepmol.base import PassThroughTransformer

from deepmol.datasets import Dataset
from deepmol.scalers.sklearn_scalers import MinMaxScaler
from deepmol.compound_featurization import *

def optimized_steps_2(trial, data: Dataset):
    mode = data.mode
    fpSize = trial.suggest_int('fpSize', 1024, 2048, step=1024)
    minPath = trial.suggest_int('minPath', 1, 3)
    maxPath = trial.suggest_int('maxPath', 5, 10)
    f1 = LayeredFingerprint(fpSize=fpSize, minPath=minPath, maxPath=maxPath, n_jobs=10)
    f2 = TwoDimensionDescriptors()

    featurizer = MixedFeaturizer([f1, f2], n_jobs=10)
    feature_selector = percentil_fs(trial, task_type = mode)
    if featurizer.__class__.__name__ == 'TwoDimensionDescriptors' or \
            featurizer.__class__.__name__ == 'All3DDescriptors':
        scaler = _get_scaler(trial)
    else:
        scaler = PassThroughTransformer()
    sk_model = hist_gradient_boosting_classifier_step(trial)

    final_steps = [('standardizer', _get_standardizer(trial, featurizer)), ('featurizer', featurizer),
                   ('scaler', scaler),
                   ('feature_selector', feature_selector), ('model', sk_model)]
    return final_steps


def optimized_steps(trial, data: Dataset):
    mode = data.mode
    multitask = True if data.n_tasks > 1 else False
    featurizer = _get_featurizer(trial, '1D')
    if featurizer.__class__.__name__ == 'TwoDimensionDescriptors' or \
            featurizer.__class__.__name__ == 'All3DDescriptors':
        scaler = _get_scaler(trial)
    else:
        scaler = PassThroughTransformer()
    feature_selector = _get_feature_selector(trial, task_type=mode, multitask=multitask)
    if mode == 'classification':
        sk_mode = 'classification_binary' if set(data.y) == {0, 1} else 'classification_multiclass'
    else:
        sk_mode = mode
    sk_model = hist_gradient_boosting_classifier_step(trial)
    if sk_model.model.__class__.__name__ == 'BernoulliNB' or sk_model.model.__class__.__name__ == 'MultinomialNB' or \
            sk_model.model.__class__.__name__ == 'ComplementNB':
        if featurizer.__class__.__name__ == 'TwoDimensionDescriptors' or \
                featurizer.__class__.__name__ == 'All3DDescriptors' or \
                featurizer.__class__.__name__ == 'Mol2Vec':
            scaler = MinMaxScaler()

    final_steps = [('standardizer', _get_standardizer(trial, featurizer)), ('featurizer', featurizer),
                   ('scaler', scaler),
                   ('feature_selector', feature_selector), ('model', sk_model)]
    return final_steps

def run():
    init_time = time.time()
    group = get_benchmark_group['Bioavailability_Ma']
    benchmark = group.get('Bioavailability_Ma')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    pipeline_name = 'bioavailability'
    storage = f'sqlite:///{pipeline_name}.db'
    general_tdc_pipeline(pipeline_name=pipeline_name, group=group, tdc_dataset_name='Bioavailability_Ma',
                         data_sample=data, seed=1234, optimizer='tpe', storage=storage, metric=roc_auc_score,
                         direction='maximize', n_trials=100, save_top_n=100, trial_timeout=60 * 60)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')

def run_optimized():
    init_time = time.time()
    group = get_benchmark_group['Bioavailability_Ma']
    benchmark = group.get('Bioavailability_Ma')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    pipeline_name = 'bioavailability'
    storage = f'sqlite:///{pipeline_name}.db'
    general_tdc_pipeline(pipeline_name=pipeline_name, group=group, tdc_dataset_name='Bioavailability_Ma',
                         data_sample=data, seed=1234, optimizer='tpe', storage=storage, metric=roc_auc_score,
                         direction='maximize', n_trials=100, save_top_n=100, trial_timeout=60 * 60, steps=optimized_steps_2)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    run_optimized()

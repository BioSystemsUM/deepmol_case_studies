import time

from deepmol.datasets import SmilesDataset
from sklearn.metrics import roc_auc_score

from dcs.pipelines.general_tdc_pipeline import general_tdc_pipeline
from dcs.utils import get_benchmark_group

import tensorflow as tf
import sys

import os

def run_sklearn():
    init_time = time.time()
    group = get_benchmark_group['HIA_Hou']
    benchmark = group.get('HIA_Hou')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    general_tdc_pipeline(pipeline_name='HIA_Hou_pipeline_sklearn/', group=group, tdc_dataset_name='HIA_Hou',
                         data_sample=data, trial_timeout=60*10, metric=roc_auc_score, direction='maximize',
                         n_trials=100, save_top_n=100, seed=123, storage='sqlite:///HIA_Hou_pipeline_sklearn.db', objective_preset='sklearn')
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')

def run_keras():
    init_time = time.time()
    group = get_benchmark_group['HIA_Hou']
    benchmark = group.get('HIA_Hou')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    general_tdc_pipeline(pipeline_name='HIA_Hou_pipeline_keras/', group=group, tdc_dataset_name='HIA_Hou',
                         data_sample=data, trial_timeout=60*10, metric=roc_auc_score, direction='maximize',
                         n_trials=100, save_top_n=100, seed=123, storage='sqlite:///HIA_Hou_pipeline_keras.db', objective_preset='keras')
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')

def run_deepchem():
    init_time = time.time()
    group = get_benchmark_group['HIA_Hou']
    benchmark = group.get('HIA_Hou')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    general_tdc_pipeline(pipeline_name='HIA_Hou_pipeline_deepchem/', group=group, tdc_dataset_name='HIA_Hou',
                         data_sample=data, trial_timeout=60*10, metric=roc_auc_score, direction='maximize',
                         n_trials=100, save_top_n=100, seed=123, storage='sqlite:///HIA_Hou_pipeline_deepchem.db', objective_preset='deepchem')
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    os.environ['PYTHONHASHSEED'] = str(123)
    os.environ['TF_DETERMINISTIC_OPS'] = 'False'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    tf.random.set_seed(123)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print('GPU detected')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        print('GPU set up correctly')

    run_deepchem()

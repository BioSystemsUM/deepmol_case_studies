import time

from deepmol.datasets import SmilesDataset

from deepmol_models.pipelines import general_tdc_pipeline
from deepmol_models.utils import get_benchmark_group
from scipy.stats import spearmanr

def spearman(x, y):
    return spearmanr(x, y)[0]

def run():
    init_time = time.time()
    group = get_benchmark_group['VDss_Lombardo']
    benchmark = group.get('VDss_Lombardo')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    pipeline_name = 'vdss'
    storage = f'sqlite:///{pipeline_name}.db'
    general_tdc_pipeline(pipeline_name=pipeline_name, group=group, tdc_dataset_name='VDss_Lombardo',
                         data_sample=data, seed=321, optimizer='tpe', storage=storage, metric=spearman,
                         direction='maximize', n_trials=100, save_top_n=100, trial_timeout=60 * 5)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    run()

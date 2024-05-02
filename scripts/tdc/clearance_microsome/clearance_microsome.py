import time

from deepmol.datasets import SmilesDataset
from tdc.benchmark_group import admet_group

from dcs.pipelines import general_tdc_pipeline
from dcs.utils import get_benchmark_group
from dcs._utils import spearman


def run():
    init_time = time.time()
    group = admet_group(path='data/')
    # group = get_benchmark_group['Clearance_Microsome_AZ']
    benchmark = group.get('Clearance_Microsome_AZ')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    pipeline_name = 'clearance_microsome'
    storage = f'sqlite:///{pipeline_name}.db'
    general_tdc_pipeline(pipeline_name=pipeline_name, group=group, tdc_dataset_name='Clearance_Microsome_AZ',
                         data_sample=data, seed=321, optimizer='tpe', storage=storage, metric=spearman,
                         direction='maximize', n_trials=100, save_top_n=100, trial_timeout=60 * 60 * 3)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    run()

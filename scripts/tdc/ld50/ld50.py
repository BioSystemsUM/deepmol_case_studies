import time

from deepmol.datasets import SmilesDataset
from sklearn.metrics import mean_absolute_error
from tdc.benchmark_group import admet_group

from deepmol_models.pipelines import general_tdc_pipeline


def run():
    init_time = time.time()
    group = admet_group(path='data/')
    #group = get_benchmark_group['LD50_Zhu']
    benchmark = group.get('LD50_Zhu')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    pipeline_name = 'ld50'
    storage = f'sqlite:///{pipeline_name}.db'
    general_tdc_pipeline(pipeline_name=pipeline_name, group=group, tdc_dataset_name='LD50_Zhu',
                         data_sample=data, seed=1234, optimizer='tpe', storage=storage, metric=mean_absolute_error,
                         direction='minimize', n_trials=100, save_top_n=100, trial_timeout=60 * 60 * 3)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    run()

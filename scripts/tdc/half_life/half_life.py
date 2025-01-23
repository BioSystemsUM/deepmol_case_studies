import time

from deepmol.datasets import SmilesDataset

from deepmol_models.pipelines import general_tdc_pipeline
from deepmol_models.utils import get_benchmark_group
from deepmol_models._utils import spearman


def run():
    init_time = time.time()
    group = get_benchmark_group['Half_Life_Obach']
    benchmark = group.get('Half_Life_Obach')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    pipeline_name = 'half_life'
    storage = f'sqlite:///{pipeline_name}.db'
    general_tdc_pipeline(pipeline_name=pipeline_name, group=group, tdc_dataset_name='Half_Life_Obach',
                         data_sample=data, seed=321, optimizer='tpe', storage=storage, metric=spearman,
                         direction='maximize', n_trials=100, save_top_n=100, trial_timeout=60 * 60)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    run()

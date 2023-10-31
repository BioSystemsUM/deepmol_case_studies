import time

from deepmol.datasets import SmilesDataset
from sklearn.metrics import roc_auc_score

from dcs.pipelines import general_tdc_pipeline
from dcs.utils import get_benchmark_group


def run():
    init_time = time.time()
    group = get_benchmark_group['Bioavailability_Ma']
    benchmark = group.get('Bioavailability_Ma')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    pipeline_name = 'bioavailability2'
    storage = f'sqlite:///{pipeline_name}/{pipeline_name}.db'
    general_tdc_pipeline(pipeline_name=pipeline_name, group=group, tdc_dataset_name='Bioavailability_Ma',
                         data_sample=data, seed=123, optimizer='tpe', storage=storage, metric=roc_auc_score,
                         direction='maximize', n_trials=5, save_top_n=5, trial_timeout=60 * 3)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    run()

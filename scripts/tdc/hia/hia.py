import time

from deepmol.datasets import SmilesDataset
from sklearn.metrics import roc_auc_score

from dcs.pipelines import general_tdc_pipeline
from dcs.utils import get_benchmark_group


def run():
    init_time = time.time()
    group = get_benchmark_group['HIA_Hou']
    benchmark = group.get('HIA_Hou')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    pipeline_name = 'hia'
    storage = f'sqlite:///{pipeline_name}.db'
    general_tdc_pipeline(pipeline_name=pipeline_name, group=group, tdc_dataset_name='HIA_Hou',
                         data_sample=data, seed=321, optimizer='tpe', storage=storage, metric=roc_auc_score,
                         direction='maximize', n_trials=100, save_top_n=100, trial_timeout=60 * 5)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    run()
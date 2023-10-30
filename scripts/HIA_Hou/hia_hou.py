import time

from deepmol.datasets import SmilesDataset

from dcs.pipelines.test_pipeline import test_pipeline
from dcs.tdc.get_tdc_data import get_benchmark_group


def run():
    init_time = time.time()
    group = get_benchmark_group['HIA_Hou']
    benchmark = group.get('HIA_Hou')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    test_pipeline(pipeline_name='HIA_Hou_pipeline/', group=group, tdc_dataset_name='HIA_Hou',
                  data_sample=data, trial_timeout=30)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    run()

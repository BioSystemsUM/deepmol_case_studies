from deepmol.datasets import SmilesDataset

from dcs.pipelines.test_pipeline import test_pipeline
from dcs.tdc.evaluate import evaluate
from dcs.tdc.get_tdc_data import get_benchmark_group


def run():
    group = get_benchmark_group['Bioavailability_Ma']
    benchmark = group.get('Bioavailability_Ma')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    bioavailability_results = test_pipeline(pipeline_name='bioavailability/', group=group,
                                            tdc_dataset_name='Bioavailability_Ma', data_sample=data)
    print(bioavailability_results)


if __name__ == '__main__':
    run()
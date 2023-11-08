import time

from deepmol.datasets import SmilesDataset
from sklearn.metrics import precision_recall_curve, auc

from dcs.pipelines import general_tdc_pipeline
from dcs.utils import get_benchmark_group


# TODO: check if works this way
def area_under_precision_recall_curve(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_pred)
    return auc(recall, precision)


def run():
    init_time = time.time()
    group = get_benchmark_group['CYP3A4_Substrate_CarbonMangels']
    benchmark = group.get('CYP3A4_Substrate_CarbonMangels')
    train_val = benchmark['train_val']
    data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    pipeline_name = 'cyp3a4_substrate'
    storage = f'sqlite:///{pipeline_name}.db'
    general_tdc_pipeline(pipeline_name=pipeline_name, group=group, tdc_dataset_name='CYP3A4_Substrate_CarbonMangels',
                         data_sample=data, seed=321, optimizer='tpe', storage=storage,
                         metric=area_under_precision_recall_curve, direction='maximize', n_trials=100, save_top_n=100,
                         trial_timeout=60 * 5)
    final_time = time.time()
    print(f'Elapsed time: {final_time - init_time}')


if __name__ == '__main__':
    run()

from deepmol.datasets import SmilesDataset
from tdc.benchmark_group import admet_group

from dcs.pipelines.test_pipeline import test_pipeline


def evaluate():
    group = admet_group(path='data/')
    benchmark = group.get('Bioavailability_Ma')

    predictions = {}
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']
    print(train_val.shape, test.shape)
    print(train_val.head())

    ## --- train your model --- ##

    train_data = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
    test_data = SmilesDataset(smiles=test['Drug'].values, ids=test['Drug_ID'].values, y=test['Y'].values)

    model = test_pipeline(train_data, test_data)

    y_pred = model.predict(test)

    predictions[name] = y_pred
    print(group.evaluate(predictions))
    # {'caco2_wang': {'mae': 0.234}}

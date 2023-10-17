from deepmol.datasets import SmilesDataset
from tdc import benchmark_group


def evaluate(pipeline_func: callable, group: benchmark_group, tdc_dataset_name: str):
    predictions_list = []

    for seed in [1, 2, 3, 4, 5]:
        benchmark = group.get(tdc_dataset_name)
        # all benchmark names in a benchmark group are stored in group.dataset_names
        predictions = {}
        name = benchmark['name']
        train_val, test = benchmark['train_val'], benchmark['test']
        train, valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

        # --------------------------------------------- #
        #  Train your model using train, valid, test    #
        #  Save test prediction in y_pred_test variable #
        # --------------------------------------------- #
        train = SmilesDataset(smiles=train['Drug'].values, ids=train['Drug_ID'].values, y=train['Y'].values)
        valid = SmilesDataset(smiles=valid['Drug'].values, ids=valid['Drug_ID'].values, y=valid['Y'].values)
        test = SmilesDataset(smiles=test['Drug'].values, ids=test['Drug_ID'].values, y=test['Y'].values)
        print(train.get_shape(), valid.get_shape(), test.get_shape())

        model = pipeline_func(train, valid, pipeline_name=f'{name}_{seed}', seed=seed)
        y_pred_test = model.predict(test)

        predictions[name] = y_pred_test
        predictions_list.append(predictions)

    results = group.evaluate_many(predictions_list)
    return results

import os

import pandas as pd

from deepmol.pipeline import Pipeline, VotingPipeline
from deepmol.datasets import SmilesDataset

from dcs.utils.tdc_benchmark_groups import get_benchmark_group
from tdc.benchmark_group import admet_group

def get_results_for_test_set_for_specific_trial(trial, dataset_path, dataset_name, tdc_dataset_name):
    pipeline_path = os.path.join(dataset_path, f'{dataset_name}', f'trial_{trial}')
    pipeline = Pipeline.load(pipeline_path)

    group = admet_group(path='data/')
    benchmark = group.get(tdc_dataset_name)
    
    name = benchmark['name']
    predictions_list = []
    for seed in [1, 2, 3, 4, 5]:
        predictions = {}
        train_val, test = benchmark['train_val'], benchmark['test']
        train_dataset = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values,
                                        y=train_val['Y'].values)
        test_set = SmilesDataset(smiles=test['Drug'].values, ids=test['Drug_ID'].values,
                                y=test['Y'].values)
        pipeline.fit(train_dataset)
        try:
            y_pred_test = pipeline.predict_proba(test_set)
            if len(y_pred_test.shape) > 1:
                y_pred_test = y_pred_test[:, 1]
        except:
            y_pred_test = pipeline.predict(test_set)
        predictions[name] = y_pred_test
        predictions_list.append(predictions)

    results = group.evaluate_many(predictions_list)
    for _, value in results.items():
        print(f'Average: {value[0]}, Std: {value[1]}')

def get_results_for_test_set(dataset_path, dataset_name, tdc_dataset_name, objective):
    """
    This function is used to get results for the test set.
    :param dataset_path: path to the dataset
    :return: None
    """
    results = os.path.join(dataset_path, f'{dataset_name}_trials.csv')
    results = pd.read_csv(results)
    if objective == 'minimize':
        results = results.sort_values(by='value', ascending=True)
    else:
        results = results.sort_values(by='value', ascending=False)
    results = results.head(10)
    if not os.path.exists(os.path.join(dataset_path, f'{dataset_name}_test_set.csv')):
        results_dataframes = pd.DataFrame(columns=['Trial', 'Average', 'Std'])
    else:
        results_dataframes = pd.read_csv(os.path.join(dataset_path, f'{dataset_name}_test_set.csv'))
    group = admet_group(path='data/')
    benchmark = group.get(tdc_dataset_name)

    print("Starting for dataset: ", dataset_name)

    print(f'Pipeline: {results.iloc[0]["number"]}, metric: {results.iloc[0]["value"]}')
    for i, row in results.iterrows():
        number = row["number"]
        value = row["value"]
        print(f'Pipeline: {number}, metric: {value}')
        pipeline_path = os.path.join(dataset_path, f'{dataset_name}', f'trial_{number}')
        pipeline = Pipeline.load(pipeline_path)
        
        name = benchmark['name']
        predictions_list = []
        for seed in [1, 2, 3, 4, 5]:
            predictions = {}
            train_val, test = benchmark['train_val'], benchmark['test']
            train_dataset = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values,
                                            y=train_val['Y'].values)
            test_set = SmilesDataset(smiles=test['Drug'].values, ids=test['Drug_ID'].values,
                                    y=test['Y'].values)
            pipeline.fit(train_dataset)
            try:
                y_pred_test = pipeline.predict_proba(test_set)
                if len(y_pred_test.shape) > 1:
                    y_pred_test = y_pred_test[:, 1]
            except:
                y_pred_test = pipeline.predict(test_set)
            predictions[name] = y_pred_test
            predictions_list.append(predictions)

        results = group.evaluate_many(predictions_list)
        for _, value in results.items():
            results_dataframes.loc[len(results_dataframes)] = [number, value[0], value[1]]
            print(f'Average: {value[0]}, Std: {value[1]}')
        results_dataframes.to_csv(os.path.join(dataset_path, f'{dataset_name}_test_set.csv'), index=False)
    
    voting_pipeline_path = os.path.join(dataset_path, f'{dataset_name}', f'voting_pipeline')
    voting_pipeline = VotingPipeline.load(voting_pipeline_path)
    name = benchmark['name']
    predictions_list = []
    for seed in [1, 2, 3, 4, 5]:
        predictions = {}
        train_val, test = benchmark['train_val'], benchmark['test']
        train_dataset = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values,
                                        y=train_val['Y'].values)
        test_set = SmilesDataset(smiles=test['Drug'].values, ids=test['Drug_ID'].values,
                                y=test['Y'].values)
        voting_pipeline.fit(train_dataset)
        try:
            y_pred_test = voting_pipeline.predict_proba(test_set)
            if len(y_pred_test.shape) > 1:
                y_pred_test = y_pred_test[:, 1]
        except:
            y_pred_test = voting_pipeline.predict(test_set)
        predictions[name] = y_pred_test
        predictions_list.append(predictions)
    results = group.evaluate_many(predictions_list)
    for _, value in results.items():
        results_dataframes.loc[len(results_dataframes)] = ['voting_pipeline', value[0], value[1]]
        print(f'Average: {value[0]}, Std: {value[1]}') 
        results_dataframes.to_csv(os.path.join(dataset_path, f'{dataset_name}_test_set.csv'), index=False)
            
if __name__ == '__main__':
    #get_results_for_test_set_for_specific_trial(74, "bioavailability", "bioavailability", tdc_dataset_name='Bioavailability_Ma')
    #get_results_for_test_set(dataset_path='cyp3a4_substrate/', dataset_name='cyp3a4_substrate', tdc_dataset_name='CYP3A4_Substrate_CarbonMangels', objective='maximize')
    #get_results_for_test_set(dataset_path='half_life/', dataset_name='half_life', tdc_dataset_name='Half_Life_Obach', objective='maximize')
    #get_results_for_test_set(dataset_path='hia/', dataset_name='hia', tdc_dataset_name='HIA_Hou', objective='maximize')
    #get_results_for_test_set(dataset_path='dili/', dataset_name='dili', tdc_dataset_name='DILI', objective='maximize')
    #get_results_for_test_set(dataset_path='herg/', dataset_name='herg', tdc_dataset_name='hERG', objective='maximize')
    
    # get_results_for_test_set(dataset_path='ld50/', dataset_name='ld50', tdc_dataset_name='LD50_Zhu', objective='minimize')
    # get_results_for_test_set(dataset_path='lipophilicity/', dataset_name='lipophilicity_stacking_regressor', tdc_dataset_name='Lipophilicity_AstraZeneca', objective='minimize')
    # get_results_for_test_set(dataset_path='pgp/', dataset_name='pgp', tdc_dataset_name='Pgp_Broccatelli', objective='maximize')
    # get_results_for_test_set(dataset_path='ppbr/', dataset_name='ppbr', tdc_dataset_name='PPBR_AZ', objective='minimize')
    # get_results_for_test_set(dataset_path='solubility/', dataset_name='solubility', tdc_dataset_name='Solubility_AqSolDB', objective='minimize')
    # get_results_for_test_set(dataset_path='vdss/', dataset_name='vdss', tdc_dataset_name='VDss_Lombardo', objective='maximize')
    # get_results_for_test_set(dataset_path='ames/', dataset_name='ames', tdc_dataset_name='AMES', objective='maximize')
    # get_results_for_test_set(dataset_path='bbb/', dataset_name='bbb', tdc_dataset_name='BBB_Martins', objective='maximize')
    get_results_for_test_set(dataset_path='bioavailability/', dataset_name='bioavailability', tdc_dataset_name='Bioavailability_Ma', objective='maximize')
    #get_results_for_test_set(dataset_path='caco/', dataset_name='caco', tdc_dataset_name='Caco2_Wang', objective='minimize')
    # get_results_for_test_set(dataset_path='clearance_hepatocyte/', dataset_name='clearance_hepatocyte', tdc_dataset_name='Clearance_Hepatocyte_AZ', objective='maximize')
    # get_results_for_test_set_for_specific_trial(78, "bbb", "bbb", tdc_dataset_name='BBB_Martins')
     

    


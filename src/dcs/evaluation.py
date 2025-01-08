from deepmol.pipeline import Pipeline, VotingPipeline
from deepmol.datasets import SmilesDataset

from tdc.benchmark_group import admet_group
import os

from dcs._utils import download_models

def _get_results(model_path, tdc_dataset_name, voting_pipeline=False):
    if not voting_pipeline:
        pipeline = Pipeline.load(model_path)
    else:
        pipeline = VotingPipeline.load(model_path)

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
    return results

available_pipelines = ["ames", "bbb", "bioavailability", "bioavailability_optimal", "caco", "clearance_hepatocyte",
                       "clearance_microsome", "hia", "pgp", "solubility", "lipophilicity", "lipophilicity_optimal",
                       "vdss", "cyp2c9", "cyp2d6", "cyp3a4", "cyp2c9_substrate", "cyp2d6_substrate", "cyp3a4_substrate",
                       "dili", "half_life", "herg", "hia", "ld50", "ppbr"]

tdc_datasets_to_pipelines = {
    "AMES": ["ames"],
    "BBB_Martins": ["bbb"],
    "Bioavailability_Ma": ["bioavailability", "bioavailability_optimal"],
    "Caco2_Wang": ["caco"],
    "Clearance_Hepatocyte_AZ": ["clearance_hepatocyte"],
    "Clearance_Microsome_AZ": ["clearance_microsome"],
    "HIA_Hou": ["hia"],
    "Pgp_Broccatelli": ["pgp"],
    "Solubility_AqSolDB": ["solubility"],
    "Lipophilicity_AstraZeneca": ["lipophilicity", "lipophilicity_optimal"],
    "VDss_Lombardo": ["vdss"],
    "CYP2C9_Veith": ["cyp2c9"],
    "CYP2D6_Veith": ["cyp2d6"],
    "CYP3A4_Veith": ["cyp3a4"],
    "CYP2C9_Substrate_CarbonMangels": ["cyp2c9_substrate"],
    "CYP2D6_Substrate_CarbonMangels": ["cyp2d6_substrate"],
    "CYP3A4_Substrate_CarbonMangels": ["cyp3a4_substrate"],
    "DILI": ["dili"],
    "Half_Life_Obach": ["half_life"],
    "hERG": ["herg"],
    "HIA_Hou": ["hia"],
    "LD50_Zhu": ["ld50"],
    "PPBR_AZ": ["ppbr"]
}



def get_results(tdc_dataset_name, pipeline=None):
    from rdkit import RDLogger

    RDLogger.DisableLog('rdApp.info')

    folder_exists = os.path.exists(os.path.join(os.path.expanduser("~"), ".deepmol_case_studies"))
    if not folder_exists:
        download_models()
    else:
        print("Models already in cache.")

    if tdc_dataset_name not in tdc_datasets_to_pipelines:
        raise ValueError(f"Dataset {tdc_dataset_name} not found in TDC datasets. Available datasets are: {tdc_datasets_to_pipelines.keys()}")
    elif pipeline is not None and pipeline not in tdc_datasets_to_pipelines[tdc_dataset_name]:
        raise ValueError(f"Pipeline {pipeline} do not correspond to dataset {tdc_dataset_name}")

    if pipeline is None:
        model_path = os.path.join(os.path.expanduser("~"), ".deepmol_case_studies", "models", tdc_datasets_to_pipelines[tdc_dataset_name][0])
    else:
        if pipeline not in available_pipelines:
            raise ValueError(f"Pipeline {pipeline} not found. Available pipelines are: {available_pipelines}")
        model_path = os.path.join(os.path.expanduser("~"), ".deepmol_case_studies", "models", pipeline)

    files = os.listdir(model_path)
    if "trial" in files[0] or "voting_pipeline" in files[0]:
        pipeline_folder_path = os.path.join(model_path, files[0])
    else:
        pipeline_folder_path = os.path.join(model_path, files[1])
        
    voting_pipeline = False
    if "voting_pipeline" in files[0]:
        voting_pipeline = True
    elif len(files) > 1 and "voting_pipeline" in files[1]:
        voting_pipeline = True
        
    return _get_results(pipeline_folder_path, tdc_dataset_name, voting_pipeline)
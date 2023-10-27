import os
import pickle
from typing import List, Tuple

from deepmol.datasets import SmilesDataset
from deepmol.loaders import CSVLoader

from dcs.pipelines.molecular_starter_prediction_pipeline import run_molecular_starter_prediction_pipeline


def prepare_data() -> List[Tuple[SmilesDataset, SmilesDataset]]:
    # read pickle
    field_task15 = ['Anthranilate',
                    'Cholesterol',
                    'GGPP',
                    'Indole.3',
                    'IPP',
                    'L.Ala',
                    'L.Arg',
                    'L.Asp',
                    'L.His',
                    'L.Lys',
                    'L.Phe',
                    'L.Pro',
                    'L.Trp',
                    'L.Tyr',
                    'Secologanin']
    data_path = "data/alkaloid_data.csv"
    dataset = CSVLoader(data_path, smiles_field='SMILES', labels_fields=field_task15, id_field='CID').create_dataset()
    cv_data = []
    for cv in range(5):
        cv_path = f"data/cv_data/cv_{cv}"
        with open(os.path.join(cv_path, f"dataset_trn_cv_{cv}.pickle"), 'rb') as f:
            train_cv_data = pickle.load(f)
        with open(os.path.join(cv_path, f"dataset_tst_cv_{cv}.pickle"), 'rb') as f:
            test_cv_data = pickle.load(f)

        train_cv_data = dataset.select(train_cv_data, inplace=False)
        test_cv_data = dataset.select(test_cv_data, inplace=False)
        cv_data.append((train_cv_data, test_cv_data))

    return cv_data


def run():
    cv_data = prepare_data()
    run_molecular_starter_prediction_pipeline(cv_data, n_trials=7, timeout=60*3, save_top_n=3)


if __name__ == "__main__":
    run()


import os
import tracemalloc
from hurry.filesize import size
import datetime
import time
import numpy as np
import pandas as pd
from deepmol.compound_featurization import *
from deepmol.scalers import *
from deepmol.feature_selection import *
from deepmol.splitters import *
from deepmol.imbalanced_learn import *
from deepmol.loaders import CSVLoader

def benchmark_splitters(dataset_path, smiles_field, label, dataset_name, ids):

    # Load data from CSV file
    loader = CSVLoader(dataset_path=dataset_path,
                    smiles_field=smiles_field,
                    labels_fields=[label],
                    id_field=ids,
                    mode='auto')
    # create the dataset
    d1 = loader.create_dataset(sep=',', header=0)

    train_dataset, validation_dataset, test_dataset = SimilaritySplitter().train_valid_test_split(dataset=d1, frac_train=0.6, frac_valid=0.2, homogenous_threshold=0.7)

    os.makedirs(os.path.join("data_for_runtime_benchmark", dataset_name), exist_ok=True)

    train_dataset.to_csv(os.path.join("data_for_runtime_benchmark", dataset_name, "train.csv"))
    validation_dataset.to_csv(os.path.join("data_for_runtime_benchmark", dataset_name, "val.csv"))
    test_dataset.to_csv(os.path.join("data_for_runtime_benchmark", dataset_name, "test.csv"))


benchmark_splitters("pgp.csv", "Drug", "Y", "pgp", "Drug_ID")
benchmark_splitters("CYP2D6_Veith.csv", "Drug", "Y", "cyp2d6", "Drug_ID")
benchmark_splitters("DEL.csv", "smiles", "true_labels_R", "del", None)



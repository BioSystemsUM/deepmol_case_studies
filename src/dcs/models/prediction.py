import os
from deepmol.loaders import CSVLoader
from deepmol.pipeline import Pipeline

from dcs._utils import download_deployed_models, download_models

import numpy as np
import pandas as pd

import pickle 


def _get_pipeline_path(model):
    folder_exists = os.path.exists(os.path.join(os.path.expanduser("~"), ".deepmol_models"))
    if not folder_exists:
        download_deployed_models()
    else:
        print("Models already in cache.")

    model_path = os.path.join(os.path.expanduser("~"), ".deepmol_models", "deployed_models", model)
    return model_path


def _make_predictions_and_sort(csv_path, smiles_field, id_field, model, mode="auto"):

    loader = CSVLoader(dataset_path=csv_path, smiles_field=smiles_field, id_field=id_field, mode=mode)
    dataset = loader.create_dataset()

    pipeline_path = _get_pipeline_path(model)
    pipeline = Pipeline.load(pipeline_path)

    pandas_dataset = pd.read_csv(csv_path)

    ids_normal_dataset = np.array(pandas_dataset.loc[:, id_field]).astype(str)
    smiles_normal_dataset = np.array(pandas_dataset.loc[:, smiles_field])

    dataset_transformed = pipeline.transform(dataset)

    predictions = pipeline.predict(dataset)

    ids_dataset_transformed = dataset_transformed.ids

    # Assuming ids_normal_dataset and ids_dataset_transformed are lists or numpy arrays
    ids_transformed = np.array(ids_dataset_transformed)  # Transformed dataset IDs

    # Find missing IDs
    missing_ids = np.setdiff1d(ids_normal_dataset, ids_transformed)

    # Create a dataframe to align predictions
    # Assuming predictions is a numpy array of shape (len(ids_transformed), num_predictions)
    num_predictions = predictions.shape[1] if len(predictions.shape) > 1 else 1
    null_predictions = np.full((len(missing_ids), num_predictions), np.nan)  # Null values for missing IDs

    # Align IDs and predictions
    aligned_ids = np.concatenate((ids_transformed, missing_ids))
    aligned_predictions = np.vstack((predictions, null_predictions))

    # Sort by the original order of ids_normal_dataset
    order_ = np.argsort(np.argsort(ids_normal_dataset))
    final_ids = aligned_ids[order_]
    final_predictions = aligned_predictions[order_]
    final_smiles_dataset = smiles_normal_dataset[order_]

    return final_ids, final_smiles_dataset, final_predictions


def _np_classifier_from_csv(csv_path, smiles_field, id_field):

    this_file_directory = os.path.dirname(os.path.abspath(__file__))
    
    final_ids, final_smiles_dataset, final_predictions = \
        _make_predictions_and_sort(csv_path, smiles_field, id_field, "np_classifier_trained", mode = ["classification"]*730)

    # Convert to a dataframe for easier manipulation
    results_df = pd.DataFrame(columns=["ID", "SMILES", "Pathways", "Superclass", "Class"])
    results_df["ID"] = final_ids
    results_df["SMILES"] = final_smiles_dataset

    labels_files = ["char2idx_path_V1.pkl", "char2idx_super_V1.pkl", "char2idx_class_V1.pkl"]

    labels = []
    labels_categories = []
    for label_file in labels_files:

        # Open and load the pickle file
        with open(os.path.join(this_file_directory, f"np_classifier/{label_file}"), "rb") as file:
            data = pickle.load(file)

        labels.extend(list(data.keys()))
        labels_categories.append(len(list(data.keys())))

    pathways_i = labels_categories[0]
    super_pathways_i = pathways_i + labels_categories[1]
    class_i = super_pathways_i + labels_categories[2]

    predictions = pd.DataFrame(final_predictions, columns=labels)
    # Transform to an array of column names with 1s
    pathways_columns_with_ones = predictions.apply(lambda row: predictions.iloc[:, :pathways_i].columns[row[:pathways_i] == 1].tolist(), axis=1)
    super_pathways_columns_with_ones = \
        predictions.apply(lambda row: predictions.iloc[:, pathways_i:super_pathways_i].columns[row[pathways_i:super_pathways_i] == 1].tolist(), axis=1)
    class_columns_with_ones = \
        predictions.apply(lambda row: predictions.iloc[:, super_pathways_i:class_i].columns[row[super_pathways_i:class_i] == 1].tolist(), axis=1)

    # Convert to a NumPy array if needed
    results_df["Pathways"] = pathways_columns_with_ones.apply(lambda x: ', '.join(x))
    results_df["Superclass"] = super_pathways_columns_with_ones.apply(lambda x: ', '.join(x))
    results_df["Class"] = class_columns_with_ones.apply(lambda x: ', '.join(x))

    return results_df
    
    

def predict_from_csv(model, csv_path, smiles_field, id_field, output_path=None):

    download_deployed_models()
    
    if model == "np_classifier":
        predictions = _np_classifier_from_csv(csv_path, smiles_field, id_field)
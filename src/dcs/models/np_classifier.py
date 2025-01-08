import os
import pickle

import pandas as pd
from dcs.models.model import PredictionModel


class NPClassifier(PredictionModel):

    model_name = "NPClassifier"
    prediction_type = "Pathway, Superclass, class"
    description = """
    This model is a reimplementation of NPClassifier described in https://pubs.acs.org/doi/10.1021/acs.jnatprod.1c00399. \n
    All credits should be given to the authors. NPClassifier performs automated structural classification of natural products. \n
    """
    features = """
    - Prediction of natural product Pathway, Superclass and Class.
    - Efficient performance on large datasets.
    """
    model = "np_classifier_trained"
    mode = ["classification"]*730

    def process_predictions(self, final_ids, final_smiles_dataset, final_predictions):
        
        this_file_directory = os.path.dirname(os.path.abspath(__file__))
    
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
    
    

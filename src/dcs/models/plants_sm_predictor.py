import numpy as np
import pandas as pd
from dcs.models.model import PredictionModel



labels_ = {
        'C00341': 'Geranyl diphosphate',
        'C01789': 'Campesterol',
        'C00078': 'Tryptophan',
        'C00049': 'L-Aspartate',
        'C00183': 'L-Valine',
        'C03506': 'Indoleglycerol phosphate',
        'C00187': 'Cholesterol',
        'C00079': 'L-Phenylalanine',
        'C00047': 'L-Lysine',
        'C01852': 'Secologanin',
        'C00407': 'L-Isoleucine',
        'C00129': 'Isopentenyl diphosphate',
        'C00235': 'Dimethylallyl diphosphate',
        'C00062': 'L-Arginine',
        'C00353': 'Geranylgeranyl diphosphate',
        'C00148': 'L-Proline',
        'C00073': 'L-Methionine',
        'C00108': 'Anthranilate',
        'C00123': 'L-Leucine',
        'C00135': 'L-Histidine',
        'C00448': 'Farnesyl diphosphate',
        'C00082': 'L-Tyrosine',
        'C00041': 'L-Alanine',
        'C00540': 'Cinnamoyl-CoA',
        'C01477': 'Apigenin',
        'C05903': 'Kaempferol',
        'C05904': 'Pelargonin',
        'C05905': 'Cyanidin',
        'C05908': 'Delphinidin',
        'C00389': 'Quercetin',
        'C01514': 'Luteolin',
        'C09762': "Liquiritigenin",
        'C00509': 'Naringenin',
        'C00223': 'p-Coumaroyl-CoA'
    }

kegg_labels = ['C00073', 'C00078', 'C00079', 'C00082', 'C00235', 'C00341', 'C00353',
              'C00448', 'C01789', 'C03506', 'C00047', 'C00108', 'C00187', 'C00148',
              'C00041', 'C00129', 'C00062', 'C01852', 'C00049', 'C00135', 'C00223',
              'C00509', 'C00540', 'C01477', 'C05903', 'C05904', 'C05905', 'C05908',
              'C09762']


def convert_predictions_into_names_model(predictions):

    labels_names = np.array([labels_[label] for label in kegg_labels])
    ones = predictions == 1
    labels_all = []
    for i, prediction in enumerate(ones):
        labels_all.append("; ".join(labels_names[prediction]))
    return labels_all

class PlantsSMPrecursorPredictor(PredictionModel):

    model_name = "Plants secondary metabolite precursors predictor"
    prediction_type = "Precursor 1; Precursor 2"
    description = """
    This model is the same model available in https://github.com/jcapels/SMPrecursorPredictor/ \n
    This model predicts plant secondary metabolite precursors. 
    """
    features = """
    - Prediction of plant secondary metabolites precursors.
    - Prediction of metabolites as precursors in metabolism such as amino-acids, terpenoids, etc...
    - Multi-label classification, so more than one (or no) precursors can be predicted. 
    """
    model = "PlantsSMPredictor"
    mode = ["classification"]*29
    results_label = "Precursors"
    download_url = "https://zenodo.org/records/14653345/files/PlantsSMPredictor.zip?download=1"


    def process_predictions(self, final_ids, final_smiles_dataset, final_predictions):
        
        # Convert to a dataframe for easier manipulation
        results_df = pd.DataFrame(columns=["ID", "SMILES", "Precursors"])
        results_df["ID"] = final_ids
        results_df["SMILES"] = final_smiles_dataset
        results_df["Precursors"] = convert_predictions_into_names_model(final_predictions)

        return results_df


        
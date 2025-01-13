
from deepmol.loaders import CSVLoader
from deepmol.datasets import SmilesDataset

from dcs.models.model import PredictionModel
import numpy as np
import pandas as pd

class MixedPredictor(PredictionModel):

    model_name = "Mixed Classifier"
    prediction_type = ""
    description = ""
    features = ""


    def __init__(self, models):
        self.models = models
        self.model_name += f" ({', '.join([model.model_name for model in self.models])})"

        for model in self.models:
            self.prediction_type += f"{model.model_name}: {model.prediction_type} \n" 
            self.description += f"Description of {model.model_name}: {model.prediction_type} \n" 
            self.features += f"{model.model_name}: \n {model.features} \n" 

        super().__init__()

    def process_predictions(self, final_ids, final_smiles_dataset, final_predictions):
        pass

    def predict_from_smi(self, smiles, output_file=None):
        results_df = self.models[0].predict_from_smi(smiles)
        for model in self.models[1:]:
            print(f"Predicting using {model.model_name} model")
            new_results = model.predict_from_smi(smiles)
            new_results = new_results.loc[:, model.results_label]
            results_df = pd.concat((results_df, new_results), axis=1)

        if output_file:
            results_df.to_csv(output_file, index=False)
        
        return results_df

    def predict_from_csv(self, csv_path, smiles_field, id_field, output_file=None):
        
        print(f"Predicting using {self.models[0].model_name} model")
        results_df = self.models[0].predict_from_csv(csv_path, smiles_field, id_field)
        for model in self.models[1:]:
            print(f"Predicting using {model.model_name} model")
            new_results = model.predict_from_csv(csv_path, smiles_field, id_field)
            new_results = new_results.loc[:, model.results_label]
            results_df = pd.concat((results_df, new_results), axis=1)

        if output_file:
            results_df.to_csv(output_file, index=False)
        
        return results_df


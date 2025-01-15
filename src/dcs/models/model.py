from abc import ABCMeta, abstractmethod
import os

import numpy as np
import pandas as pd

from dcs.models.prediction import _get_pipeline_path

from deepmol.loaders import CSVLoader
from deepmol.pipeline import Pipeline, VotingPipeline
from deepmol.datasets import SmilesDataset

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from rdkit import RDLogger

RDLogger.DisableLog('rdApp.info') 

class PredictionModel(metaclass=ABCMeta):

    model_name = ""
    prediction_type = ""
    description = """
    """
    features = """
    """
    model = ""
    mode = ""

    def __init__(self):
        import tensorflow as tf

        # Set the device to CPU
        tf.config.set_visible_devices([], 'GPU')

    def represent(self):
        console = Console()
    
        # Create a styled table for the model details
        table = Table(title=f"{self.model_name} Details", show_header=True, header_style="bold blue")
        table.add_column("Attribute", style="dim", width=20)
        table.add_column("Value", style="bold")
        table.add_row("Model Name", self.model_name)
        table.add_row("Prediction Type", self.prediction_type)
        
        # Use Text instead of Markdown for the description
        description = Text()
        description.append(f"{self.model_name} Overview\n", style="bold underline blue")
        description.append(
            self.description, style="dim",
        )

        # Key features as a panel
        features = Panel.fit(
            self.features,
            title="Key Features",
            border_style="green",
        )

        # Render the content using rich
        with console.capture() as capture:
            console.print(description)
            console.print(table)
            console.print(features)
        
        return capture.get()

    def __repr__(self):
        return self.represent()
    
    def _predict(self, dataset):

        pipeline_path = _get_pipeline_path(self.model)

        files = os.listdir(pipeline_path)

        if any(["voting_pipeline" in file for file in files]):
            pipeline = VotingPipeline.load(pipeline_path)

        else:
            pipeline = Pipeline.load(pipeline_path)

        final_predictions = pipeline.predict(dataset, return_invalid=True)


        return final_predictions

    @abstractmethod
    def process_predictions(self, final_ids, final_smiles_dataset, final_predictions):
        """
        Process predictions
        """

    def predict_from_smi(self, smiles, output_file=None):
        ids_normal_dataset = [str(i) for i in range(1, len(smiles))]
        smiles_normal_dataset = smiles

        dataset = SmilesDataset(smiles=smiles, ids=ids_normal_dataset, mode=self.mode)

        final_predictions = self._predict(dataset)

        results_df = self.process_predictions(ids_normal_dataset, smiles_normal_dataset, final_predictions)
        if output_file:
            results_df.to_csv(output_file, index=False)
        
        return results_df
    
    def predict_from_csv(self, csv_path, smiles_field, id_field, output_file=None):

        loader = CSVLoader(dataset_path=csv_path, smiles_field=smiles_field, id_field=id_field, mode=self.mode)
        dataset = loader.create_dataset()

        original_dataset = pd.read_csv(csv_path)
        ids_normal_dataset, smiles_normal_dataset = original_dataset.loc[:, id_field], original_dataset.loc[:, smiles_field]
        del original_dataset

        final_predictions = self._predict(dataset)

        results_df = self.process_predictions(ids_normal_dataset, smiles_normal_dataset, final_predictions)

        if output_file:
            results_df.to_csv(output_file, index=False)
        
        return results_df

    

        
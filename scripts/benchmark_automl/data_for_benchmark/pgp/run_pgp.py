import datetime
import os
import pickle
import time
import tracemalloc

from deepmol.datasets import SmilesDataset
from sklearn.metrics import f1_score, roc_auc_score

import os
import time

from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
import optuna
from deepmol.pipeline_optimization._utils import preset_all_models

import tensorflow as tf
from deepmol.loaders import CSVLoader

import pandas as pd

from hurry.filesize import size
from sklearn.metrics import average_precision_score

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def f1_score_macro(y_true, y_pred):

    return f1_score(y_true, y_pred, average="macro")

def run():
    tracemalloc.start()
    start = time.time()
    pipeline_name = "pgp"
    pipeline_name = pipeline_name if pipeline_name is not None else f'pipeline_{time.strftime("%Y_%m_%d-%H_%M_%S")}'
    # create the directory pipeline_name
    os.makedirs(pipeline_name, exist_ok=True)
    storage = f'sqlite:///{pipeline_name}.db'
    metric = Metric(roc_auc_score)
    pipeline = PipelineOptimization(storage=storage,
                                    study_name=pipeline_name,
                                    direction="maximize", load_if_exists=True)

    def objective_steps(trial: optuna.Trial, data):
        return preset_all_models(trial, data)

    # get splits
    current_directory = os.path.dirname(os.path.abspath(__file__))

    train_dataset = CSVLoader(dataset_path=os.path.join(current_directory, "train.csv"), 
                              smiles_field="smiles",labels_fields=["Y"]).create_dataset()
    
    validation_dataset = CSVLoader(dataset_path=os.path.join(current_directory, "val.csv"), 
                              smiles_field="smiles",labels_fields=["Y"]).create_dataset()


    steps = objective_steps

    pipeline.optimize(train_dataset = train_dataset, test_dataset=validation_dataset, objective_steps=steps, n_trials=100, save_top_n=5,
                      trial_timeout=60*60*2, metric=metric, data=train_dataset)
    
    print(pipeline.trials_dataframe())
    print(f"Best trial: {pipeline.best_trial}")
    print(f"Best score: {pipeline.best_value}")
    print(f"Best params: {pipeline.best_params}")
    # save trials_dataframe to csv
    pipeline.trials_dataframe().to_csv(f'{pipeline_name}_trials.csv')

    end = time.time()
    print("Time spent: ", end - start)
    print("Memory needed: ", tracemalloc.get_traced_memory())

    results = pd.DataFrame()
    results = pd.concat((results, 
                            pd.DataFrame({
                                            "time": [str(datetime.timedelta(seconds=end - start))], 
                                            "memory": [size(int(tracemalloc.get_traced_memory()[1]))]})), 
                                            ignore_index=True, axis=0)
    tracemalloc.stop()

    results.to_csv(os.path.join(current_directory,"runtime.csv"), index=False)

def test_best_pipeline():

    from deepmol.pipeline import Pipeline, VotingPipeline

    current_directory = os.path.dirname(os.path.abspath(__file__))

    train_dataset = CSVLoader(dataset_path=os.path.join(current_directory, "train_valid.csv"), 
                              smiles_field="smiles",labels_fields=["Y"]).create_dataset()
    
    test_dataset = CSVLoader(dataset_path=os.path.join(current_directory, "test.csv"), 
                              smiles_field="smiles",labels_fields=["Y"]).create_dataset()
    
    pipeline = VotingPipeline.load("pgp/voting_pipeline")

    pipeline.fit(train_dataset)

    predictions = pipeline.predict_proba(test_dataset)

    with open("deepmol_predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

if __name__ == '__main__':
    #run()
    test_best_pipeline()

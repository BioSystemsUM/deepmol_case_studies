# Start with the imports.
import sklearn
import optuna
import datetime
import os
import time
import tracemalloc
import pandas as pd

from hurry.filesize import size


from optunaz.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
)
from optunaz.datareader import Dataset
from optunaz.descriptors import  *
from optunaz.config.optconfig import *

# Prepare hyperparameter optimization configuration.
config = OptimizationConfig(
    data=Dataset(
        input_column="smiles",
        response_column="Y",
        training_dataset_file="train_valid.csv",
    ),
    descriptors=[
        Avalon.new(),
        ECFP.new(),
        ECFP_counts.new(),
        PathFP.new(),
        MACCS_keys.new(),
        UnscaledPhyschemDescriptors.new(),
        UnscaledJazzyDescriptors.new(),
        SmilesFromFile.new()

    ],
    algorithms=[
        ChemPropClassifier.new(epochs=4),
        RandomForestClassifier.new(),
        PRFClassifier.new(),
        AdaBoostClassifier.new(),
        KNeighborsClassifier.new(),
        SVC.new(),
        ChemPropHyperoptClassifier.new(),
        # Mapie.new()<
    ],
    settings=OptimizationConfig.Settings(
        mode=ModelMode.CLASSIFICATION,
        cross_validation=1,
        scoring="f1",
        n_trials=100,
        random_seed=42,
        direction=OptimizationDirection.MAXIMIZATION,
    ),
)

tracemalloc.start()
start = time.time()
pipeline_name = "pgp"
study = optimize(config, study_name="my_study")
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

current_directory = os.path.dirname(os.path.abspath(__file__))
results.to_csv(os.path.join(current_directory, "run_qsartuna_runtime.csv"), index=False)

# Get the best Trial from the Study and make a Build (Training) configuration for it.
buildconfig = buildconfig_best(study)
with open("best_config.txt", "w") as f:
    f.write(str(buildconfig.__dict__))

# Build (re-Train) and save the best model.
build_best(buildconfig, "best.pkl")

# Build (Train) and save the model on the merged train+test data.
build_merged(buildconfig, "merged.pkl")
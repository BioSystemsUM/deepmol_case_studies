
import pandas as pd

dataset = pd.read_csv("../data/np_classifier_dataset_augmented.csv")

from deepmol.datasets import SmilesDataset


dataset = SmilesDataset(smiles=dataset.SMILES.values, # only mandatory argument, a list of SMILES strings
                          mols=None,
                          ids=dataset.key.values,
                          X=None,
                          feature_names=None,
                          y=dataset.iloc[:,2:],
                          label_names=dataset.columns[2:],
                          mode='multilabel')

from deepmol.pipeline import Pipeline
from deepmol.compound_featurization import NPClassifierFP

import tensorflow as tf
from deepmol.models import KerasModel

def set_gpu(gpu_ids_list):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            gpus_used = [gpus[i] for i in gpu_ids_list]
            tf.config.set_visible_devices(gpus_used, 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

set_gpu([])

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

def top_k_categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

def model_build(): # num = number of categories
    input_f = layers.Input(shape=(6144,))
    # input_b = layers.Input(shape=(4096,))
    # input_fp = layers.Concatenate()([input_f,input_b])
    
    X = layers.Dense(2048, activation = 'relu')(input_f)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(3072, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.BatchNormalization()(X)
    X = layers.Dense(1536, activation = 'relu')(X)
    X = layers.Dropout(0.2)(X)
    output = layers.Dense(730, activation = 'sigmoid')(X)
    model = keras.Model(inputs = [input_f], outputs = output)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),loss=['binary_crossentropy'])
    return model

pipeline = Pipeline(steps=[("np fingerprint", NPClassifierFP()), 
                ("model", KerasModel(model_builder=model_build(), batch_size=128, epochs=100,
                                     mode="multilabel"))], path="np_classifier_trained").fit(dataset)

pipeline.save()
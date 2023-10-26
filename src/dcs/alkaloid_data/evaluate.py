from typing import List, Tuple

import numpy as np
import pandas as pd
from deepmol.datasets import SmilesDataset
from deepmol.pipeline import Pipeline
from sklearn.metrics import f1_score


def evaluate(pipeline: Pipeline, cv_data: List[Tuple[SmilesDataset, SmilesDataset]]):
    """
    Evaluate a pipeline on a given dataset.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline to evaluate.
    cv_data : list of tuples
        List of tuples (train, test) containing the training, validation and test data.
    """
    f1_scores = []

    for train, test in cv_data:
        model = pipeline.fit(train, test)
        y_pred_test = model.predict(test)
        y_test = test.y
        f1_score_ = f1_score(y_test, y_pred_test, average='macro')
        f1_scores.append(f1_score_)

    mean_f1_score = np.mean(f1_scores)
    std_f1_score = np.std(f1_scores)
    return mean_f1_score, std_f1_score


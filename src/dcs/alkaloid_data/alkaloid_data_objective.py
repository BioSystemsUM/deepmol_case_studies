import os
import shutil
from typing import List, Tuple

import optuna
from deepmol.datasets import SmilesDataset
from deepmol.metrics import Metric
from deepmol.pipeline import Pipeline
from deepmol.pipeline_optimization.objective_wrapper import Objective
from optuna import Trial, Study
from timeout_decorator import timeout_decorator

from dcs.alkaloid_data.evaluate import evaluate


class MolecularStartersObjective(Objective):

    def __init__(self, objective_steps: callable, study: Study, direction, train_dataset, test_dataset, metric: Metric,
                 save_top_n: int, trial_timeout: int, **kwargs):

        """
        Wrapper for the objective function of the pipeline optimization.
        It creates and saves pipelines for each trial and evaluates them on the test dataset.

        Parameters
        ----------
        objective_steps : callable
            Function that returns the steps of the pipeline for a given trial.
        study : optuna.study.Study
            Study object that stores the optimization history.
        direction : str or optuna.study.StudyDirection
            Direction of the optimization (minimize or maximize).
        cv_data : List[Tuple[SmilesDataset, SmilesDataset]]
            List of tuples of training and validation datasets.
        metric : deepmol.metrics.Metric
            Metric used for evaluating the pipeline.
        save_top_n : int
            Number of best pipelines to save.
        trial_timeout : int
            Timeout for each trial in seconds.
        **kwargs
            Additional keyword arguments passed to the objective_steps function.

        """
        super().__init__(objective_steps, study, direction, metric, save_top_n)
        self.trial_timeout = trial_timeout
        self.cv_data = kwargs.pop('cv_data')
        self.kwargs = kwargs

    def __call__(self, trial: Trial):
        try:
            @timeout_decorator.timeout(self.trial_timeout, timeout_exception=optuna.TrialPruned)
            def run_with_timeout():
                trial_id = str(trial.number)
                path = os.path.join(self.save_dir, f'trial_{trial_id}')
                data = self.cv_data[0][1]
                pipeline = Pipeline(steps=self.objective_steps(trial, data=data, **self.kwargs), path=path)
                mean_f1_score, std_f1_score = evaluate(pipeline, self.cv_data)
                score = mean_f1_score - std_f1_score

                best_scores = self.study.user_attrs['best_scores']

                min_score = min(best_scores.values()) if len(best_scores) > 0 else float('inf')
                max_score = max(best_scores.values()) if len(best_scores) > 0 else float('-inf')
                update_score = (self.direction == 'maximize' and score > min_score) or (
                        self.direction == 'minimize' and score < max_score)

                if len(best_scores) < self.save_top_n or update_score:
                    pipeline.save()
                    best_scores.update({trial_id: score})

                    if len(best_scores) > self.save_top_n:
                        if self.direction == 'maximize':
                            min_score_id = min(best_scores, key=best_scores.get)
                            del best_scores[min_score_id]
                            shutil.rmtree(os.path.join(self.save_dir, f'trial_{min_score_id}'))
                        else:
                            max_score_id = max(best_scores, key=best_scores.get)
                            del best_scores[max_score_id]
                            shutil.rmtree(os.path.join(self.save_dir, f'trial_{max_score_id}'))

                self.study.set_user_attr('best_scores', best_scores)
                return score

            return run_with_timeout()
        except ValueError as e:
            print(e)
            return float('inf') if self.direction == 'minimize' else float('-inf')

        except Exception as e:
            print(e)
        return float('inf') if self.direction == 'minimize' else float('-inf')

import os
import shutil

import optuna
import timeout_decorator
from deepmol.datasets import SmilesDataset
from deepmol.pipeline import Pipeline
from deepmol.pipeline_optimization.objective_wrapper import Objective
from optuna import Trial
from tdc import benchmark_group


class TDCObjective(Objective):

    def __init__(self, objective_steps, study, direction, train, test, metric, save_top_n, trial_timeout, **kwargs):
        super().__init__(objective_steps, study, direction, metric, save_top_n)
        self.group = kwargs.pop('group')
        self.tdc_dataset_name = kwargs.pop('tdc_dataset_name')
        self.trial_timeout = trial_timeout
        self.kwargs = kwargs

    def __call__(self, trial: Trial):
        try:
            @timeout_decorator.timeout(self.trial_timeout, timeout_exception=optuna.TrialPruned)
            def run_with_timeout():
                trial_id = str(trial.number)
                path = os.path.join(self.save_dir, f'trial_{trial_id}')
                scores = []
                predictions_list = []
                for seed in [1, 2, 3, 4, 5]:
                    benchmark = self.group.get(self.tdc_dataset_name)
                    # all benchmark names in a benchmark group are stored in group.dataset_names
                    predictions = {}
                    name = benchmark['name']
                    train_val, test = benchmark['train_val'], benchmark['test']
                    #train, valid = self.group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

                    # --------------------------------------------- #
                    #  Train your model using train, valid, test    #
                    #  Save test prediction in y_pred_test variable #
                    # --------------------------------------------- #
                    train_dataset = SmilesDataset(smiles=train_val['Drug'].values, ids=train_val['Drug_ID'].values, y=train_val['Y'].values)
                    #valid_dataset = SmilesDataset(smiles=valid['Drug'].values, ids=valid['Drug_ID'].values, y=valid['Y'].values)
                    test_dataset = SmilesDataset(smiles=test['Drug'].values, ids=test['Drug_ID'].values, y=test['Y'].values)
                    print(train_dataset.get_shape(), test_dataset.get_shape())

                    pipeline = Pipeline(steps=self.objective_steps(trial, **self.kwargs), path=path)
                    pipeline.fit(train_dataset)
                    scores.append(pipeline.evaluate(test_dataset, [self.metric])[0][self.metric.name])
                    y_pred_test = pipeline.predict(test_dataset)

                    predictions[name] = y_pred_test
                    predictions_list.append(predictions)

                results = self.group.evaluate_many(predictions_list)
                score = sum(scores) / len(scores)
                print(f'Average score: {score}')
                print(f'Average results: {results}')
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
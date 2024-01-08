import os
import shutil

import optuna
import timeout_decorator
from deepmol.pipeline import Pipeline
from deepmol.pipeline_optimization.objective_wrapper import Objective
from optuna import Trial


class TDCObjective(Objective):

    def __init__(self, objective_steps, study, direction, save_top_n, trial_timeout, **kwargs):
        super().__init__(objective_steps, study, direction, save_top_n)
        self.group = kwargs.pop('group')
        self.tdc_dataset_name = kwargs.pop('tdc_dataset_name')
        self.trial_timeout = trial_timeout
        self.metric = kwargs.pop('metric')
        self.splits = kwargs.pop('splits')
        self.kwargs = kwargs

    def __call__(self, trial: Trial):
        try:
            @timeout_decorator.timeout(self.trial_timeout, timeout_exception=optuna.TrialPruned)
            def run_with_timeout():
                trial_id = str(trial.number)
                path = os.path.join(self.save_dir, f'trial_{trial_id}')
                scores = []
                predictions_list = []
                benchmark = self.group.get(self.tdc_dataset_name)
                name = benchmark['name']
                for train_dataset, valid_dataset, test_dataset in zip(*self.splits):
                    # all benchmark names in a benchmark group are stored in group.dataset_names
                    predictions = {}

                    # --------------------------------------------- #
                    #  Train your model using train, valid, test    #
                    #  Save test prediction in y_pred_test variable #
                    # --------------------------------------------- #
                    pipeline = Pipeline(steps=self.objective_steps(trial, **self.kwargs), path=path)
                    if pipeline.steps[-1][1].__class__.__name__ == 'KerasModel':
                        pipeline.fit(train_dataset, validation_dataset=valid_dataset)
                    else:
                        pipeline.fit(train_dataset)
                    scores.append(pipeline.evaluate(valid_dataset, [self.metric])[0][self.metric.name])
                    try:
                        y_pred_test = pipeline.predict_proba(test_dataset)
                        if len(y_pred_test.shape) > 1:
                            y_pred_test = y_pred_test[:, 1]
                    except:
                        y_pred_test = pipeline.predict(test_dataset)

                    predictions[name] = y_pred_test
                    predictions_list.append(predictions)

                results = self.group.evaluate_many(predictions_list)
                # save results to a file
                with open(os.path.join(self.save_dir, f'tdc_test_set_results.txt'), 'a+') as f:
                    for _, value in results.items():
                        f.write(f'{trial_id},{value[0]},{value[1]}\n')
                score = sum(scores) / len(scores)
                print(f'Trial {trial_id} finished:')
                print(f'Average score: {score}')
                print(f'Average results: {results}')
                print(f'Parameters: {trial.params}')
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

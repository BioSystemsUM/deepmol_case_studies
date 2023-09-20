from deepmol.metrics import Metric
from deepmol.pipeline_optimization import PipelineOptimization
import optuna
from sklearn.metrics import roc_auc_score


def test_pipeline(train, valid):
    evo_sampler = optuna.samplers.NSGAIISampler(population_size=10, mutation_prob=0.1,
                                                crossover=None, crossover_prob=0.9,
                                                swapping_prob=0.5, seed=None, constraints_func=None)
    # AUROC
    metric = Metric(roc_auc_score)
    pipeline = PipelineOptimization(sampler=evo_sampler,
                                    study_name='test_pipeline',
                                    direction='maximize')
    pipeline.optimize(train, valid, objective_steps='all', metric=metric, n_trials=10, save_top_n=3, data=train)
    return pipeline

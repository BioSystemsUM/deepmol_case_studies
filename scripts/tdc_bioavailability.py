from dcs.pipelines.test_pipeline import test_pipeline
from dcs.tdc.evaluate import evaluate
from dcs.tdc.get_tdc_data import get_benchmark_group


def run():
    group = get_benchmark_group['Bioavailability_Ma']
    bioavailability_results = evaluate(pipeline_func=test_pipeline, group=group, tdc_dataset_name='Bioavailability_Ma')
    print(bioavailability_results)


if __name__ == '__main__':
    run()

from dcs.pipelines.molecular_starter_prediction_pipeline import run_molecular_starter_prediction_pipeline


def prepare_data():
    pass


def run():
    cv_data = prepare_data()
    run_molecular_starter_prediction_pipeline(cv_data)
import click
import mlflow
import os
from datetime import datetime


def run(entrypoint, parameters=None):
    print(f"Launching step: {entrypoint} with parameters: {parameters}")
    step = mlflow.run('.', entry_point=entrypoint, parameters=parameters, use_conda=False)
    return mlflow.tracking.MlflowClient().get_run(step.run_id)


def create_exp_env(experiment_name, directory_name):
    experiment_id = mlflow.get_experiment_by_name(experiment_name)
    if not experiment_id:
        experiment_id = mlflow.create_experiment(experiment_name, directory_name)
    else:
        experiment_id = experiment_id.experiment_id
    print("Experiment id: " + str(experiment_id))
    return experiment_id


@click.command()
@click.option("--ph", default=1, type=int)
@click.option("--cp", default=1, type=int)
@click.option("--register_model_name", default=None)
def workflow(ph, cp, register_model_name):
    mlflow.set_tracking_uri('http://localhost:5000')

    experiment_name = "churn_experiment"
    now = datetime.now()
    artifact_uri = "s3://hbg-bi-landing-test/mlflow/"
    directory_name = f"PH_{ph}_CP_{cp}_on_{now.day}_{now.month}_at_{now.hour}_{now.minute}"
    artifact_directory = artifact_uri + directory_name
    print(artifact_directory)
    experiment_id = create_exp_env(experiment_name, artifact_directory)
    experiment_id = experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name="main") as mlrun:

        print("Starting workflow")
        now = datetime.now()
        artifact_location_name = f"PH_{ph}_CP_{cp}_on_{now.day}_{now.month}_at_{now.hour}_{now.minute}"
        print(artifact_location_name)

        # Process data step
        process_data = run("process_data_client_churn")

        # Model training step
        run("model_client_churn", parameters={"data_path": "df_pickles/transformed",
                                              "run_id": process_data.info.run_id,
                                              "ph": ph, "cp": cp, "register_model_name": register_model_name})

        print("Workflow finished")


if __name__ == "__main__":
    workflow()

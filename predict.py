import mlflow
import os
import pandas as pd
from mlflow.tracking import MlflowClient

important_variables = [
        "FF_AMOUNT_OVERRIDE_EUR_SUM_L1S",
        "FF_CALLS_AVG_CASE_AGING_MEAN_L13S",
        "FF_GROSS_SALES_SUM_L13S",
        "FF_GROSS_SALES_SUM_L1S",
        "FF_ONE_OM_SUM_L13S",
        "FF_ONE_OM_SUM_L1S",
        "FF_MARKUP_MEAN_L13S",
        "FF_BOOKINGCANCELLATION_SUM_L13S",
        "FF_VALUATIONS_TOTAL_SUM_L13S",
        "FF_SEARCHES_TOTAL_SUM_L13S",
        "FF_SEARCHES_TOTAL_SUM_L1S",
    ]

if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:5000')

    with mlflow.start_run() as mlrun:
        # get data uri
        client = MlflowClient()
        model_uri = client.get_model_version_download_uri(name="churn_model", version="1")
        print(f"Model URI: {model_uri}")


        project_path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "load/")
        data_path = "df_pickles/transformed"
        path_scenarios_pickle = os.path.join(project_path, f"{data_path}/df_transform_scenarios_client_churn.pickle")
        df = pd.read_pickle(path_scenarios_pickle)

        # model_saved_path = os.path.join(project_path, "model/")

        # model_uri = "models:/churn_model/1"
        model = mlflow.lightgbm.load_model(model_uri)
        # mlflow.log_artifacts(model_saved_path)
        prediction = model.predict(df[important_variables])

        print(f"Prediction: {prediction}")

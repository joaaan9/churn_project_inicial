import sys

import click
import pandas as pd
import os

from mlflow.tracking import MlflowClient

import util.snowflake as sf
from functions_utils import function_utils
import mlflow


def model_x_S(df, ph, cp, minumim_value_to_consider_not_churn=0.005):
    df["CHURN_LxS"] = df[f"FF_GROSS_SALES_SUM_L{cp}S"] <= minumim_value_to_consider_not_churn
    # df["CHURN_L13SO"] = df["FF_GROSS_COMMERCIAL_COST_SUM_L13SO"] <= minumim_value_to_consider_not_churn
    df = df[~df["CHURN_LxS"]]

    features_fields = []
    targets_fields = []
    for element in df.columns.to_list():
        if element[:3] == "FF_":
            features_fields.append(element)
        else:
            targets_fields.append(element)

    feat_dev, target_dev, feat_test, target_test = function_utils.split_data(
        df, f"CHURN_PH_{ph}_CP_{cp}S", targets_fields, 0.3
    )

    # Looking the most interesting variables
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

    feat_dev = feat_dev[important_variables]
    feat_test = feat_test[important_variables]

    correlated_variables = function_utils.find_correlated_variables(df[important_variables], 0.8)

    # mdl = function_utils.mdl_fit(feat_dev, target_dev, categoricals = [], params = mdl_params)
    mdl = function_utils.mdl_fit(feat_dev, target_dev, categoricals=[])

    pred_dev = function_utils.mdl_pred(mdl, feat_dev, target_dev)
    pred_test = function_utils.mdl_pred(mdl, feat_test, target_test)
    pred_all = function_utils.mdl_pred(mdl, df[important_variables], df[f"CHURN_PH_{ph}_CP_{cp}S"])

    # Roc AUC Curve
    function_utils.print_roc_auc(
        [
            {"prediction": pred_dev, "curve_name": "TRAIN"},
            {"prediction": pred_test, "curve_name": "TEST"},
            {"prediction": pred_all, "curve_name": "ALL"},
        ],
        f"PH_{ph}_CP_{cp} - Client Churn - Roc curve",
    )
    # Feature Importance
    plt_feat_imp = function_utils.print_feature_importance(mdl, title=f"PH_{ph}_CP_{cp} - Client Churn - Feature Importance")
    return df, mdl, important_variables, pred_dev, pred_test, correlated_variables, plt_feat_imp


@click.command()
@click.option("--data_path", default="df_pickles/transformed")
@click.option("--run_id", default="c74728cd1a8e472895007d7701c3cd84")
@click.option("--ph", default=1, type=int)
@click.option("--cp", default=1, type=int)
@click.option("--register_model_name", default=None)
def main(data_path, run_id, ph, cp, register_model_name):
    mlflow.set_tracking_uri('http://localhost:5000')
    with mlflow.start_run() as mlrun:
        # ------------------ Load data ------------------
        minumim_value_to_consider_not_churn = 0.005
        project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))), "load/")
        path_today_pickle = os.path.join(project_path, f"{data_path}/df_transform_today_client_churn.pickle")
        path_scenarios_pickle = os.path.join(project_path, f"{data_path}/df_transform_scenarios_client_churn.pickle")

        client = MlflowClient()
        client.download_artifacts(run_id, data_path, project_path)
        df = pd.read_pickle(path_scenarios_pickle)

        # check mlflow paths
        artifact_path = mlrun.info.artifact_uri
        print(f"Artifact path for mlflow: {artifact_path}")

        # get ph and cp variable as parameters
        if ph > 4 or ph < 1:
            ph = 1
        if cp > 4 or cp < 1:
            cp = 1

        print(f"PH: {ph} and CP: {cp}")
        mlflow.set_tag("PH", ph)
        mlflow.set_tag("CP", cp)

        # Running the model
        df, mdl, important_variables, pred_dev, pred_test, corr, plt= model_x_S(df, ph=ph, cp=cp)

        # log feature_importance as artifact
        artifact_images_directory = "images"
        output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))), "output")
        feature_importance_file = "feature_importance.png"
        feat_imp_path = os.path.join(output_path, feature_importance_file)
        plt.savefig(feat_imp_path)
        mlflow.log_artifact(feat_imp_path, artifact_path=artifact_images_directory)

        # register model only if passed as parameter
        if not register_model_name or register_model_name == "None":
            print("Model is not registered")
            mlflow.lightgbm.log_model(mdl, "model")
        else:
            print("Registering model...")
            mlflow.lightgbm.log_model(mdl, "model", registered_model_name=register_model_name)

        # Loading the last window
        # @TODO revisar que datos leer today o scenarios
        df_today = pd.read_pickle(path_today_pickle)
        df_today = df_today.copy()
        df_today["CHURN_L1S"] = df_today["FF_GROSS_SALES_SUM_L1S"] <= minumim_value_to_consider_not_churn
        df_today["CHURN_L2S"] = df_today["FF_GROSS_SALES_SUM_L2S"] <= minumim_value_to_consider_not_churn
        df_today["CHURN_L3S"] = df_today["FF_GROSS_SALES_SUM_L3S"] <= minumim_value_to_consider_not_churn
        df_today["CHURN_L4S"] = df_today["FF_GROSS_SALES_SUM_L4S"] <= minumim_value_to_consider_not_churn

        # To see the distribution of the clients
        plt = function_utils.distribution(df_today)

        # log distribution as mlflow artifact
        dist_file = "distribution.png"
        dist_path = os.path.join(output_path, dist_file)
        plt.savefig(dist_path)
        mlflow.log_artifact(dist_path, artifact_path=artifact_images_directory)

        # Testing the last window. By the moment we exclude the clients churned in the last 4 slots
        df_today = df_today[~df_today["CHURN_L1S"]]
        pred_today = function_utils.mdl_pred(mdl, df_today[important_variables], df_today.CHURN_PH_1_CP_1S)

        plt, roc_auc = function_utils.print_roc_auc(
            [
                {"prediction": pred_dev, "curve_name": "Dev"},
                {"prediction": pred_today, "curve_name": "Today"},
            ],
            "Client Churn - Roc curve",
        )

        # log roc plot as mlflow artifact @TODO log plot
        roc_file = "roc_curve.png"
        roc_path = os.path.join(output_path, roc_file)
        plt.savefig(roc_path)
        mlflow.log_artifact(roc_path, artifact_path=artifact_images_directory)

        # log roc_auc as mlflow metric
        mlflow.log_metric("roc_auc", roc_auc)

        # function_utils.print_confusion_matrix(pred_today_1, title=f"{1}S - Test set", threshold=0.6)
        plt, cm = function_utils.print_confusion_matrix(pred_today, title=f"{1}S - Test set", threshold=0.6)

        # log metrics extracted from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tn + tp) / (tn + tp + fn + fp)
        positive_accuracy = tp / (tp + fp)
        negative_accuracy = tn / (tn + fn)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("positive_accuracy", positive_accuracy)
        mlflow.log_metric("negative_accuracy", negative_accuracy)

        # log confusion matrix plot as mlflow artifact
        conf_matrix_file = "confusion_matrix.png"
        conf_matrix_path = os.path.join(output_path, conf_matrix_file)
        plt.savefig(conf_matrix_path)
        mlflow.log_artifact(conf_matrix_path, artifact_path=artifact_images_directory)

        m_score, std_scores = function_utils.print_cross_validation(
            mdl, df_today[important_variables], df_today.CHURN_PH_4_CP_4S, cv=6, title="Cross Validation"
        )

        # log cross validation metrics
        mlflow.log_metric("m_score", m_score)
        mlflow.log_metric("std_scores", std_scores)



if __name__ == "__main__":
    main()

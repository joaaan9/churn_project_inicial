from project.functions_utils import function_utils
import pandas as pd
import os
import project.util.snowflake as sf


def model_x_S(df, ph, cp, minumim_value_to_consider_not_churn=0.005):
    df["CHURN_LxS"] = df[f"FF_GROSS_COMMERCIAL_COST_SUM_L{cp}S"] <= minumim_value_to_consider_not_churn
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
    feat_dev = feat_dev[features_fields]
    feat_test = feat_test[features_fields]
    mdl = function_utils.mdl_fit(
        feat_dev,
        target_dev,
        categoricals=[
            "FF_REGION_MAX_L1S",
            "FF_HOTEL_CATEGORY_GROUP_MAX_L1S",
        ],
    )

    # Filtering the most interesting columns
    feature_imp = pd.DataFrame(
        sorted(zip(mdl.feature_importances_, feat_dev.columns)), columns=["Value", "Feature"]
    ).sort_values(by="Value", ascending=False)
    top_feat_imp = list(feature_imp.head(40)["Feature"])
    correlated_variables = function_utils.find_correlated_variables(df[top_feat_imp], 0.8)
    not_correlated_feat_imp = [i for i in top_feat_imp if i not in correlated_variables["variable_2"].unique()]
    important_variables = not_correlated_feat_imp[0:15]
    important_variables = important_variables + [
        "FF_REGION_MAX_L1S",
        "FF_HOTEL_CATEGORY_GROUP_MAX_L1S",
    ]
    important_variables = list(dict.fromkeys(important_variables))

    feat_dev = feat_dev[important_variables]
    feat_test = feat_test[important_variables]
    mdl = function_utils.mdl_fit(
        feat_dev,
        target_dev,
        categoricals=[
            "FF_REGION_MAX_L1S",
            "FF_HOTEL_CATEGORY_GROUP_MAX_L1S",
        ],
    )

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
        f"PH_{ph}_CP_{cp} - Hotel Churn - Roc curve",
    )
    # Feature Importance
    function_utils.print_feature_importance(mdl, title=f"PH_{ph}_CP_{cp} - Hotel Churn - Feature Importance")
    return df, mdl, important_variables, pred_dev, pred_test, correlated_variables


if __name__ == "__main__":

    # ------------------ Load data ------------------
    sn = sf.Snowflake()
    minumim_value_to_consider_not_churn = 0.005

    df = pd.read_pickle(os.path.join(os.getcwd(), "output/df_transform_scenarios_hotel_churn.pickle"))
    # Here exclude all the hotels with some GnD in any moment
    df = df[df["FF_HAS_SOME_GND_MAX_L1S"] == 0]

    # Running the 4 models
    df_4, mdl_4, important_variables_4, pred_dev_4, pred_test_4, corr_4 = model_x_S(df, ph=4, cp=4)
    df_3, mdl_3, important_variables_3, pred_dev_3, pred_test_3, corr_3 = model_x_S(df, ph=3, cp=4)
    df_2, mdl_2, important_variables_2, pred_dev_2, pred_test_2, corr_2 = model_x_S(df, ph=2, cp=4)
    df_1, mdl_1, important_variables_1, pred_dev_1, pred_test_1, corr_1 = model_x_S(df, ph=1, cp=4)

    # Loading the last window
    df_today = pd.read_pickle(os.path.join(os.getcwd(), "output/df_transform_today_hotel_churn.pickle"))
    df_today = df_today[df_today["FF_HAS_SOME_GND_MAX_L1S"] == 0]
    df_today["CHURN_L1S"] = df_today["FF_GROSS_COMMERCIAL_COST_SUM_L1S"] <= minumim_value_to_consider_not_churn
    df_today["CHURN_L2S"] = df_today["FF_GROSS_COMMERCIAL_COST_SUM_L2S"] <= minumim_value_to_consider_not_churn
    df_today["CHURN_L3S"] = df_today["FF_GROSS_COMMERCIAL_COST_SUM_L3S"] <= minumim_value_to_consider_not_churn
    df_today["CHURN_L4S"] = df_today["FF_GROSS_COMMERCIAL_COST_SUM_L4S"] <= minumim_value_to_consider_not_churn
    # df_today["CHURN_L13SO"] = df_today["FF_GROSS_COMMERCIAL_COST_SUM_L13SO"] <= minumim_value_to_consider_not_churn

    # To see the distribution of the hotels
    function_utils.distribution(df_today)

    # Testing the last window. By the moment we exclude the hotels churned in the last 4 slots
    df_today = df_today[~df_today["CHURN_L4S"]]
    pred_today_4 = function_utils.mdl_pred(mdl_4, df_today[important_variables_4], df_today.CHURN_PH_4_CP_4S)
    pred_today_3 = function_utils.mdl_pred(mdl_3, df_today[important_variables_3], df_today.CHURN_PH_3_CP_4S)
    pred_today_2 = function_utils.mdl_pred(mdl_2, df_today[important_variables_2], df_today.CHURN_PH_2_CP_4S)
    pred_today_1 = function_utils.mdl_pred(mdl_1, df_today[important_variables_1], df_today.CHURN_PH_1_CP_4S)

    function_utils.print_roc_auc(
        [
            {"prediction": pred_dev_4, "curve_name": "Dev 4"},
            {"prediction": pred_today_4, "curve_name": "Today 4"},
            {"prediction": pred_dev_3, "curve_name": "Dev 3"},
            {"prediction": pred_today_3, "curve_name": "Today 3"},
            # {"prediction": pred_dev_2, "curve_name": "Dev 2"},
            # {"prediction": pred_today_2, "curve_name": "Today 2"},
            # {"prediction": pred_dev_1, "curve_name": "Dev 1"},
            # {"prediction": pred_today_1, "curve_name": "Today 1"},
        ],
        "Hotel Churn - Roc curve",
    )

    function_utils.print_confusion_matrix(pred_today_1, title=f"{1}S - Test set", threshold=0.6)

    function_utils.print_cross_validation(
        mdl_4, df_today[important_variables_4], df_today.CHURN_PH_4_CP_4S, cv=6, title="4S - Cross Validation"
    )
    function_utils.print_cross_validation(
        mdl_3, df_today[important_variables_3], df_today.CHURN_PH_3_CP_4S, cv=6, title="3S - Cross Validation"
    )
    function_utils.print_cross_validation(
        mdl_2, df_today[important_variables_2], df_today.CHURN_PH_2_CP_4S, cv=6, title="2S - Cross Validation"
    )
    function_utils.print_cross_validation(
        mdl_1, df_today[important_variables_1], df_today.CHURN_PH_1_CP_4S, cv=6, title="1S - Cross Validation"
    )

    save_results = False
    if save_results:
        # To save all the dataframe of dev & test
        sn.to_sql(df, "hotel_churn_trading")

        # To save the df_today results with the most important features
        df_final = pd.merge(df_today, pred_today_4[["proba"]], left_index=True, right_index=True).rename(
            columns={"proba": "PROBA_4S"}
        )
        df_final = pd.merge(df_final, pred_today_3[["proba"]], left_index=True, right_index=True).rename(
            columns={"proba": "PROBA_3S"}
        )
        df_final = pd.merge(df_final, pred_today_2[["proba"]], left_index=True, right_index=True).rename(
            columns={"proba": "PROBA_2S"}
        )
        df_final = pd.merge(df_final, pred_today_1[["proba"]], left_index=True, right_index=True).rename(
            columns={"proba": "PROBA_1S"}
        )

        df_final = df_final.rename(
            columns={
                "CHURN_PH_4_CP_4S": "CHURN_4S",
                "CHURN_PH_3_CP_4S": "CHURN_3S",
                "CHURN_PH_2_CP_4S": "CHURN_2S",
                "CHURN_PH_1_CP_4S": "CHURN_1S",
            }
        )

        list_features_importants = (
            [
                "PK",
                "SCENARIO_DATE",
                # "CHURN_L13SO",
                "FF_ONE_OM_SUM_L13S",
                "CHURN_4S",
                "CHURN_3S",
                "CHURN_2S",
                "CHURN_1S",
                "PROBA_4S",
                "PROBA_3S",
                "PROBA_2S",
                "PROBA_1S",
            ]
            + important_variables_4
            + important_variables_3
            + important_variables_2
            + important_variables_1
        )
        list_features_importants = list(dict.fromkeys(list_features_importants))

        upload_to_sql = df_final[list_features_importants].copy()
        upload_to_sql["RANK"] = upload_to_sql["FF_ONE_OM_SUM_L13S"].rank(method="max", ascending=False)

        sn.to_sql(upload_to_sql, "hotel_churn_prediction")

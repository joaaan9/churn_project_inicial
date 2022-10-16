# matplotlib.use("TkAgg")
# import matplotlib

import matplotlib.pyplot as plt

import project.util.snowflake as sf
from snowflake.connector.pandas_tools import pd_writer

# from pandas_profiling import ProfileReport
from sklearn.model_selection import cross_val_score, RandomizedSearchCV  # , train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve

from numpy import mean
from numpy import std

import boruta
import pandas as pd
import os
import heapq


# def load_data(sn):
#     df = sn.query("SELECT * FROM HBG_BI.SANDBOX_ANALYTICS.CHURN_MODELLING_TRADING")
#     return df
#
#
# def load_data_more_scenarios(sn, months=18, last_months=2, max_date=date(2022, 3, 23), n_windows=True):
#
#     if n_windows:
#         l_months = [day.strftime("%Y-%m-%d") for day in rrule(MONTHLY, dtstart=date(2021, 1, 1), until=max_date)]
#     else:
#         l_months = [max_date]
#     df_all = pd.DataFrame()
#     loading_query = open(os.path.join(os.getcwd(), "project/loadind_table.sql"), "r")
#     loading_query = loading_query.read()
#     for m in l_months:
#         q = loading_query.format(months=months, last_months=last_months, m=m)
#         df = sn.query(q)
#
#         df_all = pd.concat([df, df_all], ignore_index=True)
#     return df_all


# def create_profile_report(df, filename, title, minimal=True):
#     print("Creating profile report: %s.html" % filename)
#     profile = ProfileReport(df.reset_index(drop=True), title=title, minimal=minimal)
#     profile.to_file("%s.html" % filename)


def remove_outliers(df):
    df_copy = df.copy()

    fields_to_remove_outliers_top = [
        "GROSS_SALES_L12M",
    ]
    percentiles_top = {}
    for f in fields_to_remove_outliers_top:
        percentiles_top[f] = df_copy[f].quantile(0.998)

    fields_to_remove_outliers_bottom = [
        "GROSS_SALES_L12M",
    ]
    percentiles_bottom = {}
    for f in fields_to_remove_outliers_bottom:
        percentiles_bottom[f] = df_copy[f].quantile(0.55)

    upper_end_obs = len(df_copy[df_copy[f] >= percentiles_top[f]])
    upper_end_clients = df_copy[df_copy[f] >= percentiles_top[f]]["PK"].nunique()
    print(f"Upper End: Removing {upper_end_obs} observations, involving {upper_end_clients} clients")
    check_removal_effect(df_copy[df_copy[f] >= percentiles_top[f]], "2")

    for f in fields_to_remove_outliers_top:
        df_copy = df_copy[df_copy[f] < percentiles_top[f]]

    check_dataset_status(df_copy, "2. Removed upper end")

    lower_end_obs = len(df_copy[df_copy[f] <= percentiles_bottom[f]])
    lower_end_clients = df_copy[df_copy[f] <= percentiles_bottom[f]]["PK"].nunique()
    print(f"Lower End: Removing {lower_end_obs} observations, involving {lower_end_clients} clients")
    check_removal_effect(df_copy[df_copy[f] <= percentiles_bottom[f]], "3")

    for f in fields_to_remove_outliers_bottom:
        df_copy = df_copy[df_copy[f] > percentiles_bottom[f]]

    check_dataset_status(df_copy, "3. Removed lower end")

    return df_copy


# def balance_data(df):
#     number_not_change = int(df[df["CHURN_VARIATION"].isin(["C_L2M__NOT_C", "NOT_C_L2M__C"])].shape[0] / 2)
#     # df_shuffle = df.sample(frac=1, random_state=42).copy()
#     df_shuffle = df.sample(frac=1).copy()
#     balanced = (
#         df_shuffle[df_shuffle["CHURN_VARIATION"] == "C_L2M__C"].
#           iloc[:number_not_change, :].copy().reset_index(drop=True)
#     )
#     balanced = balanced.append(
#         df_shuffle[df_shuffle["CHURN_VARIATION"] == "NOT_C_L2M__NOT_C"].iloc[:number_not_change, :], ignore_index=True
#     ).copy()
#     balanced = balanced.append(
#         df_shuffle[df_shuffle["CHURN_VARIATION"].isin(["C_L2M__NOT_C", "NOT_C_L2M__C"])], ignore_index=True
#     ).copy()
#
#     return balanced


# def split_data(df_raw, target, excluded_fields, percentage=0.3):
#
#     df = df_raw.loc[:, [i for i in df_raw.columns if i not in excluded_fields]].copy()
#     feat_dev, feat_eval, target_dev, target_eval = train_test_split(
#         df.loc[:, df.columns != target], df[target], test_size=percentage, random_state=42, shuffle=True
#     )
#
#     feat_train, feat_test, target_train, target_test = train_test_split(
#         feat_dev, target_dev, test_size=percentage, random_state=42, shuffle=True
#     )
#
#     return feat_eval, target_eval, feat_train, feat_test, target_train, target_test, feat_dev, target_dev


def split_data(df, target, excluded_fields, percentage=0.3):

    # Use two most recent months as test set, everything else as dev set
    month_list = list(df.SCENARIO_DATE.unique())
    test_months = heapq.nlargest(2, month_list)
    dev_months = [month for month in month_list if month not in test_months]

    df_dev = df[df.SCENARIO_DATE.isin(dev_months)]
    df_test = df[df.SCENARIO_DATE.isin(test_months)]

    target_dev = df_dev[target]
    target_test = df_test[target]

    feat_dev = df_dev.loc[:, [i for i in df_dev.columns if i not in excluded_fields + [target]]].copy()
    feat_test = df_test.loc[:, [i for i in df_test.columns if i not in excluded_fields + [target]]].copy()

    return feat_dev, target_dev, feat_test, target_test


def find_important_variables(feat, target):
    """
    Perform feature selection via Boruta using a predefined, simple model.
    :param feat: dataframe of features
    :param target: series of target variables
    :return: A list of variables deemed useful by boruta
    """

    model_for_boruta = LGBMClassifier(
        # A simplified model with minimal tuning
        learning_rate=0.08,
        n_estimators=200,
        num_leaves=30,
        min_data_in_leaf=20,
        max_depth=6,
        lambda_l1=0.5,
        lambda_l2=0.5,
        feature_fraction=0.6,
        max_bin=300,
        seed=42,
    )

    # Define Boruta feature selection method
    feat_selector = boruta.BorutaPy(model_for_boruta, n_estimators="auto")
    feat_selector.fit(feat.values, target)

    return feat.columns[feat_selector.support_]


def find_correlated_variables(feat, cutoff):
    """
    Find highly correlated variables to facilitate variable selection process
    :param feat: dataframe
    :param cutoff:
    :return: A list of correlated variables.
    """
    cor = feat.corr()
    c1 = cor.stack().sort_values(ascending=False).drop_duplicates()
    high_cor = c1[c1.values != 1]
    high_cor = high_cor[high_cor > cutoff]

    # Translate result into dataframe
    df_high_cor = pd.DataFrame(high_cor).reset_index()
    df_high_cor.columns = ["variable_1", "variable_2", "correlation"]

    return df_high_cor


def mdl_find_params(feat, target, n_iter=10, cv=5, used_features=None):
    # mdl = LGBMClassifier(reg_lambda=100,reg_alpha=100)
    mdl = LGBMClassifier()

    if used_features is None:
        used_features = feat.columns

    search_params = {
        "n_estimators": [60, 80, 100],
        "max_depth": [5, 6, 7],
        "learning_rate": [0.04, 0.05, 0.07],
        "min_data_in_leaf": [20, 25, 30],
        "max_bin": [100, 200, 300],
        "feature_fraction": [0.35, 0.45, 0.55],
        "num_leaves": [15, 20, 30],
        "lambda_l1": [0, 0.25, 0.5, 1],
        "lambda_l2": [0, 0.25, 0.5, 1],
        "bagging_fraction": [0.6, 0.75, 1],
        "bagging_freq": [3, 5],
        "class_weight": ["balanced", None],
    }

    grid = RandomizedSearchCV(mdl, search_params, n_iter=n_iter, cv=cv, random_state=42)
    grid.fit(feat.loc[:, used_features], target)

    return grid.best_params_


def mdl_fit(feat, target, params, used_features=None):
    # mdl = LGBMClassifier(reg_lambda=100,reg_alpha=100)
    mdl = LGBMClassifier(**params)

    if used_features is None:
        used_features = feat.columns

    mdl.fit(feat.loc[:, used_features], target)

    return mdl


def mdl_pred(mdl, feat, target, target_col=None):

    if not isinstance(target, pd.DataFrame):
        res = target.to_frame("target")
    else:
        res = target[target_col].to_frame("target")

    res["proba"] = mdl.predict_proba(feat[mdl.feature_name_])[:, 1]
    res["pred_target"] = mdl.predict(feat[mdl.feature_name_])

    return res


def print_roc_auc(listdict, title):
    plt.figure()
    lw = 2
    plt.plot([0, 1], [0, 1], color=plt.cm.tab10.colors[0], lw=lw, linestyle="--")

    for n, i in enumerate(listdict):

        fpr, tpr, _ = roc_curve(i["prediction"]["target"], i["prediction"]["proba"])
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=plt.cm.tab10.colors[n + 1],
            lw=lw,
            label="ROC %s (AUC = %0.2f)" % (i["curve_name"], roc_auc),
        )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")

    plt.show()


def print_feature_importance(mdl, title="Feature importances"):
    importances = pd.Series(mdl.feature_importances_, index=mdl.feature_name_)
    importances = importances.sort_values(ascending=False)

    fig, ax = plt.subplots()
    importances.plot.bar(ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def print_confusion_matrix(prediction, title, threshold=0.5):
    prediction["binarized_boolean"] = prediction["proba"] >= threshold

    cm = confusion_matrix(prediction["target"], prediction["binarized_boolean"])
    _ = ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{title} - {threshold=}")
    plt.show()


def print_cross_validation(mdl, feat, target, cv, title):
    scores = cross_val_score(mdl, feat, target, cv=cv)
    print(scores)
    m_scores = round(mean(scores), 3)
    std_scores = round(std(scores), 3)
    print(title + " Accuracy: %.3f (%.3f)" % (m_scores, std_scores))

    plt.figure()
    plt.hist(scores, bins=10, density=1)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("Cross Validation" + "({}):   Mean ".format(title) + str(m_scores) + "   STD: " + str(std_scores))
    plt.show()


def check_dataset_status(df, situation):
    print(f"{situation}: Observations: {len(df)}; Clients: {df.PK.nunique()}")


def check_removal_effect(df, situation):
    print(f"{situation}. These observations has a churn probability of {round(len(df[df.CHURN])/len(df), 2)}")


if __name__ == "__main__":

    # ------------------ Load data ------------------

    sn = sf.Snowflake()

    # df_all_etl = load_data_more_scenarios(sn, max_date=date(2022, 2, 28), n_windows=True)
    # df_all_last = load_data_more_scenarios(sn, max_date=date(2022, 2, 28), n_windows=False)
    # df_all_etl.groupby("ETL").count()["PK"]
    # df_all_last.groupby("etl").count()["PK"]

    # df_all_etl.groupby("etl").count()["pk"]
    # df_all_etl.to_csv("output/data.csv", index=False)
    # df_all_last.to_csv("output/df_all_etl.csv", index=False)
    # df = df_all_etl[~df_all_etl["churn_l2m"]]
    # df = load_data()
    # df = df_all_etl
    # df = df_all_last

    # df = pd.read_csv(os.path.join(os.getcwd(), "output/data.csv"))
    # df_all_last = pd.read_csv(os.path.join(os.getcwd(), "output/df_all_etl.csv"))

    df = pd.read_pickle(os.path.join(os.getcwd(), "output/df_transform_scenarios.pickle"))
    df["GROSS_SALES_L1M"] = df["GROSS_SALES_L1M"].astype(float)
    df["GROSS_SALES_L12M"] = df["GROSS_SALES_L12M"].astype(float)

    # df = df.head(10000) # Speed up computation

    check_dataset_status(df, "1. Just loaded data")

    # ------------------ Remove outliers and already-churned observations ------------------
    # print(f"{len(df)=}, {df.PK.nunique()=}, {min(df.RN_L12M)=}, {max(df.RN_L12M)=}")

    df = remove_outliers(df)

    # Remove churned observations
    # Using 0.005 instead of 0 in case of floating point problems
    print(f"Churned previous month: Removing {len(df[df.GROSS_SALES_L1M < 0.005])} observations")
    check_removal_effect(df[df.GROSS_SALES_L1M < 0.005], "4")
    check_dataset_status(df, "4. Removing previous month churn observation")

    df = df[df.GROSS_SALES_L1M >= 0.005]
    check_removal_effect(df, "Up to this point....")

    # ------------------ Remove highly correlated variables ------------------

    # correlated_variables = find_correlated_variables(df, 0.8)  # Note that nothing is processed in-place here.

    df.drop(
        [
            "CALLS_SUM_ATTENDED_L1M",
            "CALLS_SUM_ATTENDED_L12M",
            "RES_DSWEB_OUT_OF_SCOPE_L12M",
            "RES_DSWEB_OUT_OF_SCOPE_L1M",
            "RES_DS_OUT_OF_SCOPE_L12M",
            "CALLS_NUM_OTHER_CASES_L12M",
            "CALLS_NUM_OTHER_CASES_L1M",
            "CALLS_SUM_AVRG_AHT_L1M",
            "CALLS_SUM_AVRG_AHT_L12M",
            "TOTALITY_SERVICE_L1M",
            "TOTALITY_SERVICE_L12M",
            "CASES_OPS_STATUS_SOLVED_L12M",
            "CASES_OPS_STATUS_SOLVED_L1M",
            "BOOKINGS_TOTAL_L1M",
            "BOOKINGS_TOTAL_L12M",
            "CALLS_SUM_CALLS_L1M",
            "CALLS_SUM_CALLS_L12M",
            "ACCOMOCATION_L1M",
            "ACCOMOCATION_L12M",
            "RN_L1M",
            "RN_L12M",
        ],
        axis=1,
        inplace=True,
    )

    # ------------------ Make certain variables proportional ------------------

    lookup_dct = {
        "BOOKINGCANCELLATION_L1M": "GROSS_SALES_L1M",
        "BOOKINGS_WITH_CASES_OPS_L1M": "GROSS_SALES_L1M",
        "BOOKINGS_WITH_CASES_SALES_L1M": "GROSS_SALES_L1M",
        "BOOKINGS_WITH_CASES_FINANCE_L1M": "GROSS_SALES_L1M",
        "BOOKOUT_L1M": "GROSS_SALES_L1M",
        "CALLS_TOTAL_NUM_CASES_L1M": "GROSS_SALES_L1M",
        # "VALUATIONS_TOTAL_L1M": "SEARCHES_TOTAL_L1M",
        "SEARCHES_FAILED_L1M": "SEARCHES_TOTAL_L1M",
        "BOOKINGCANCELLATION_L12M": "GROSS_SALES_L12M",
        "BOOKINGS_WITH_CASES_OPS_L12M": "GROSS_SALES_L12M",
        "BOOKINGS_WITH_CASES_SALES_L12M": "GROSS_SALES_L12M",
        "BOOKINGS_WITH_CASES_FINANCE_L12M": "GROSS_SALES_L12M",
        "BOOKOUT_L12M": "GROSS_SALES_L12M",
        "CALLS_TOTAL_NUM_CASES_L12M": "GROSS_SALES_L12M",
        "VALUATIONS_TOTAL_L12M": "SEARCHES_TOTAL_L12M",
        "SEARCHES_FAILED_L12M": "SEARCHES_TOTAL_L12M",
    }

    for numerator, denominator in lookup_dct.items():
        df[f"P_{numerator}"] = df[numerator] / [max(1, i) for i in df[denominator]]

    # ------------------ Pandas profiling ------------------

    # create_profile_report(df, 'profile report clean', 'profile_report_clean')
    # create_profile_report(df, 'profile report ext', 'profile_report_ext', minimal=False)

    # ------------------ Balancing observations ------------------

    # balanced = balance_data(df)
    # agg_balanced = balanced.groupby(['CHURN_L2M', 'CHURN', 'CHURN_VARIATION']).count()['PK']

    # agg_balanced
    # balanced = df

    # ------------------ Splitting data ------------------

    feat_dev, target_dev, feat_test, target_test = split_data(
        df, "CHURN", ["PK", "CHURN_L2M", "CHURN_VARIATION", "ETL", "SCENARIO_DATE", "GS_TARGET"], 0.3
    )

    # ------------------ Variable selection ------------------

    # Doesn't like NAs, using 0 as placeholders which may not be appropriate as we add new variables
    # important_variables = find_important_variables(feat_dev.fillna(value=0), target_dev)
    important_variables = [
        "AMOUNT_OVERRIDE_EUR_L1M",
        "CALLS_AVG_CASE_AGING_L12M",
        "GROSS_SALES_L12M",
        "GROSS_SALES_L1M",
        "MARKUP_L12M",
        "P_BOOKINGCANCELLATION_L12M",
        "P_VALUATIONS_TOTAL_L12M",
        "SEARCHES_TOTAL_L12M",
        "SEARCHES_TOTAL_L1M",
        "VALUATIONS_TOTAL_L12M",
        "VALUATIONS_TOTAL_L1M",
    ]

    feat_dev = feat_dev[important_variables]
    feat_test = feat_test[important_variables]

    # ------------------ Modelling ------------------

    # mdl_params = mdl_find_params(feat_dev, target_dev)
    mdl_params = {
        "num_leaves": 20,
        "n_estimators": 80,
        "min_data_in_leaf": 30,
        "max_depth": 6,
        "max_bin": 300,
        "learning_rate": 0.04,
        "lambda_l2": 1,
        "lambda_l1": 0.5,
        "feature_fraction": 0.45,
        "class_weight": None,
        "bagging_freq": 3,
        "bagging_fraction": 0.6,
    }

    mdl = mdl_fit(feat_dev, target_dev, mdl_params)

    pred_dev = mdl_pred(mdl, feat_dev, target_dev)
    pred_test = mdl_pred(mdl, feat_test, target_test)
    pred_all = mdl_pred(mdl, df[important_variables], df.CHURN)

    # ------------------ Post-Modelling analysis  ------------------

    # AUC
    print_roc_auc(
        [
            {"prediction": pred_dev, "curve_name": "TRAIN"},
            {"prediction": pred_test, "curve_name": "TEST"},
            {"prediction": pred_all, "curve_name": "ALL"},
        ],
        "Roc curve",
    )

    # Feature Importance
    print_feature_importance(mdl, title="Feature Importance")

    # Confusion matrix
    print_confusion_matrix(pred_all, title="All clients", threshold=0.7)
    # print_confusion_matrix(pred_test, title="Test set", threshold=0.7)

    # In-depth analysis
    df = pd.merge(df, pred_all[["proba"]], left_index=True, right_index=True)
    df.plot.scatter("GS_TARGET", "proba", logx=True)

    df_correct_prediction = df[(df.proba >= 0.75) & df.CHURN]

    # ------------------------- Upload results to snowflake -------------------------

    upload = df[["PK", "SCENARIO_DATE", "CHURN", "proba"]]

    upload = upload.head(10000)
    upload.SCENARIO_DATE = upload.SCENARIO_DATE.astype(str)  # to_sql hates datetime64[ns]

    cursor = sn.cnx.cursor()
    cursor.execute("DROP TABLE hbg_datascience.sandbox_analytics.fct_churn_prediction")
    cursor.close()
    upload.to_sql(
        "FCT_CHURN_PREDICTION", sn.create_engine(), if_exists="replace", index=False, chunksize=16000, method=pd_writer
    )

    # ------------------------- TEST (ignore this)

    # This shows that class_weight='balanced' does not improve results

    mdl = LGBMClassifier()
    used_features = feat_dev.columns

    # Please first run mdl_params = mdl_find_params(feat_dev, target_dev) in the modelling section
    search_params = {key: [value] for key, value in mdl_params.items()}

    # Check class_weight

    search_params["class_weight"] = ["balanced", None]

    grid = RandomizedSearchCV(mdl, search_params, n_iter=2, cv=5, random_state=42)
    grid.fit(feat_dev.loc[:, used_features], target_dev)

    grid.best_params_
    grid.cv_results_

    # Check unbalance parameter instead

    del search_params["class_weight"]

    search_params["unbalance"] = [True, False]

    grid = RandomizedSearchCV(mdl, search_params, n_iter=2, cv=5, random_state=42)
    grid.fit(feat_dev.loc[:, used_features], target_dev)

    grid.best_params_
    grid.cv_results_

    del search_params["unbalance"]

    # feat_eval, target_eval, feat_train, feat_test, target_train, target_test, feat_dev, target_dev = split_data(
    #                                                                                       balanced, 'CHURN',
    #                                                                                      ['PK', 'CHURN_L2M',
    #                                                                                    'CHURN_VARIATION','ETL'], 0.3)

    # feature_list = pd.Series(mdl.feature_importances_, index=mdl.feature_name_).sort_values(ascending=False)
    # for i in range(len(feature_list)):
    #     mdl = mdl_fit(feat_train, target_train, used_features=feature_list.index[:i+1].to_list())
    #
    #     pred_train = mdl_pred(mdl, feat_train, target_train)
    #     pred_test = mdl_pred(mdl, feat_test, target_test)
    #     pred_eval = mdl_pred(mdl, feat_eval, target_eval)
    #     pred_all = mdl_pred(mdl, df, df['CHURN'])
    #
    #     print_roc_auc([{'prediction': pred_train, 'curve_name': 'TRAIN'},
    #                    {'prediction': pred_test, 'curve_name': 'TEST'},
    #                    {'prediction': pred_eval, 'curve_name': 'EVAL'},
    #                    {'prediction': pred_all, 'curve_name': 'ALL'}],
    #                   'Roc curve with %s features' % (str(i+1)))
    #     pred_train = mdl_pred(mdl, feat_train, target_train)

    # Probably outdated?
    # results = pd.DataFrame()
    # for i in ["NOT_C_L2M__C", "NOT_C_L2M__NOT_C"]:
    #     aux = df[df["CHURN_VARIATION"] == i]
    #     aux_pred = mdl_pred(mdl, aux, aux["CHURN"])
    #     cm = confusion_matrix(aux_pred["pred_target"], aux_pred["target"])
    #     results = results.append(
    #         {
    #             "Combination": i,
    #             "False - False": cm[0, 0],
    #             "False - True": cm[0, 1],
    #             "True - False": cm[1, 0],
    #             "True - True": cm[1, 1],
    #         },
    #         ignore_index=True,
    #     )
    #
    # f_f = results["False - False"].sum()
    # f_t = results["False - True"].sum()
    # t_f = results["True - False"].sum()
    # t_t = results["True - True"].sum()
    # results = results.append(
    #     {"Combination": "ALL", "False - False": f_f, "False - True": f_t, "True - False": t_f, "True - True": t_t},
    #     ignore_index=True,
    # )
    #
    # w_f_t = results.loc[results["Combination"] == "NOT_C_L2M__C", "False - True"][0]
    # w_f_f = results.loc[results["Combination"] == "NOT_C_L2M__NOT_C", "True - False"][1]
    # results["Error"] = [w_f_t, w_f_f, w_f_t + w_f_f]
    # results["ALL"] = results.loc[:, results.columns[1:6]].sum(axis=1)
    #
    # results["ERR"] = results["Error"] / results["ALL"]
    #
    # print_confusion_matrix(aux_pred, title="aux")

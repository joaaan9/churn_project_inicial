from pandas_profiling import ProfileReport
import heapq
import boruta
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from functools import reduce
import os
from transform_data import TransformData


class function_utils:
    @staticmethod
    def get_list_features(features):
        # Metric, operacion, meses, agg
        # RN, sum, [1, 2, 18], "only"
        # MARKUP, avg, [2, 5], "agg"
        list_features = list(map(lambda val: [(x, val[0], val[2]) for x in val[1]], features))
        list_features = reduce(lambda a, b: a + b, list_features)
        list_features = reduce(
            lambda grp, val: grp[val[0], val[2]].append(val[1]) or grp, list_features, defaultdict(list)
        )
        return list_features

    @staticmethod
    def load_data(sn, in_pickle=False, model="HOTEL"):
        if model == "HOTEL":
            aux_name = "hotel_churn"
            output_name = "df_raw_hotel_churn.pickle"
            snowflake_table = "SELECT * FROM HBG_DATASCIENCE.SANDBOX_ANALYTICS.FCT_HOTEL_CHURN_AGG"
            path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "output", "df_raw_hotel_churn.pickle")
        else:
            aux_name = "client_churn"
            output_name = "df_raw_client_churn.pickle"
            snowflake_table = "SELECT * FROM HBG_DATASCIENCE.SANDBOX_ANALYTICS.FCT_CHURN_MODELLING_DATA_AGG"
            path = os.path.join(
                (os.path.dirname(os.path.dirname(os.path.abspath("__file__")))),
                "output",
                output_name,
            )

        if not in_pickle:
            df = sn.query(snowflake_table)
            df.to_pickle(path)
        # Or, load existing data
        else:
            df = pd.read_pickle(path)
        return df

    @staticmethod
    def save_data(df, name, model="HOTEL"):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))), "output")
        print(path)
        path = os.path.join(path, f"{name}.pickle")
        df.to_pickle(path)

    @staticmethod
    def load_data_more_scenarios(df, features, start_time=2, end_time=8, len=1, model="HOTEL", target_combinations=[]):

        if model == "HOTEL":
            target_metric = "GROSS_COMMERCIAL_COST"
        else:
            target_metric = "GROSS_SALES"

        df_times = [i for i in range(start_time, end_time)]
        scenarios = []

        for m in df_times:
            df_copy = df.loc[(df["TIME_SLOT"] >= m - len), :].copy()
            df_copy["TIME_SLOT"] = df_copy["TIME_SLOT"] - m + len
            result = TransformData(
                df_copy,
                break_date=len,
                model=model,
                features_definition_months=features,
                target_combinations=target_combinations,
            ).transform_data()
            result["SCENARIO_DATE"] = m
            scenarios.append(result)
            print("Done scenario: " + str(m))
            print(result.agg({f"{target_metric}_TARGET_PH_1_CP_1": "sum", f"FF_{target_metric}_SUM_L1S": "sum"}))

        df_all = pd.concat(scenarios, ignore_index=True)
        return df_all

    @staticmethod
    def get_all_targets_combinations():

        options = [
            {"PH": 4, "CP": 4},
            {"PH": 4, "CP": 3},
            {"PH": 4, "CP": 2},
            {"PH": 4, "CP": 1},
            {"PH": 3, "CP": 4},
            {"PH": 3, "CP": 3},
            {"PH": 3, "CP": 2},
            {"PH": 3, "CP": 1},
            {"PH": 2, "CP": 4},
            {"PH": 2, "CP": 3},
            {"PH": 2, "CP": 2},
            {"PH": 2, "CP": 1},
            {"PH": 1, "CP": 4},
            {"PH": 1, "CP": 3},
            {"PH": 1, "CP": 2},
            {"PH": 1, "CP": 1},
        ]
        combinations = []
        for o in options:
            # In the tupple we can see
            # First position: the prediction horizont
            # Second position: the churn period
            # Third position: the first slot considered included
            # Fourth position: the last slot considered included
            combinations.append(
                {"PH": o["PH"], "CP": o["CP"], "F_SLOT": 4 - o["PH"] + o["CP"] - 1, "L_SLOT": 4 - o["PH"]}
            )

        return combinations

    @staticmethod
    def create_profile_report(df, filename, title, minimal=True):
        """
        It's used in the model to se the profiling.
        Run using:
        create_profile_report(df,"hotel_churn_profile","hotel_churn_profile",minimal=True)
        create_profile_report(df,"hotel_churn_profile_extend","hotel_churn_profile_extend",minimal=False)
        """
        print("Creating profile report: %s.html" % filename)
        profile = ProfileReport(df.reset_index(drop=True), title=title, minimal=minimal)
        profile.to_file("%s.html" % filename)

    @staticmethod
    def split_data(df, target, excluded_fields, percentage=0.3):
        """
        To split the data with dev and test
        Use two most recent slots as test set, everything else as dev set
        """
        month_list = list(df.SCENARIO_DATE.unique())
        test_months = heapq.nsmallest(2, month_list)
        dev_months = [month for month in month_list if month not in test_months]

        df_dev = df[df.SCENARIO_DATE.isin(dev_months)]
        df_test = df[df.SCENARIO_DATE.isin(test_months)]

        target_dev = df_dev[target]
        target_test = df_test[target]

        feat_dev = df_dev.loc[:, [i for i in df_dev.columns if i not in excluded_fields + [target]]].copy()
        feat_test = df_test.loc[:, [i for i in df_test.columns if i not in excluded_fields + [target]]].copy()

        return feat_dev, target_dev, feat_test, target_test

    @staticmethod
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

    @staticmethod
    def mdl_find_params(feat, target, n_iter=10, cv=5, used_features=None):

        mdl = LGBMClassifier()

        if used_features is None:
            used_features = feat.columns

        search_params = {
            "n_estimators": [60, 80, 100],
            "max_depth": [5, 6, 7],
            "learning_rate": [0.03, 0.04, 0.05, 0.07],
            "min_data_in_leaf": [10, 20, 25, 30, 35, 40],
            "max_bin": [100, 200, 300, 400],
            "feature_fraction": [0.35, 0.45, 0.55],
            "num_leaves": [15, 20, 30, 40, 100, 150],
            "lambda_l1": [0, 0.1, 0.2, 0.25, 0.5, 1],
            "lambda_l2": [0, 0.25, 0.5, 1],
            "bagging_fraction": [0.5, 0.6, 0.75, 1],
            "bagging_freq": [3, 5, 7],
            "class_weight": ["balanced", None],
        }

        grid = RandomizedSearchCV(mdl, search_params, n_iter=n_iter, cv=cv, random_state=42)
        grid.fit(feat.loc[:, used_features], target)

        return grid.best_params_

    @staticmethod
    def find_correlated_variables(feat, cutoff):
        """
        Find highly correlated variables to facilitate variable selection process
        :param feat: dataframe
        :param cutoff: the threshold to cut
        :return: A list of correlated variables.
        It's sorted at the end of function to keep always first the most important features
        """
        cor = feat.corr()
        c1 = cor.stack().drop_duplicates()
        high_cor = c1[c1.values != 1]
        high_cor = high_cor[high_cor > cutoff]

        # Translate result into dataframe
        df_high_cor = pd.DataFrame(high_cor).reset_index()
        df_high_cor.columns = ["variable_1", "variable_2", "correlation"]
        df_high_cor = df_high_cor.sort_values(by="correlation", ascending=False)

        return df_high_cor

    @staticmethod
    def mdl_fit(feat, target, categoricals, params=None, used_features=None):
        if params is None:
            mdl = LGBMClassifier()
        else:
            mdl = LGBMClassifier(**params)

        if used_features is None:
            used_features = feat.columns

        mdl.fit(feat.loc[:, used_features], target) # @TODO revisar categorical?? , categorical_feature=categoricals)
        return mdl

    @staticmethod
    def mdl_pred(mdl, feat, target, target_col=None):

        if not isinstance(target, pd.DataFrame):
            res = target.to_frame("target")
        else:
            res = target[target_col].to_frame("target")

        res["proba"] = mdl.predict_proba(feat[mdl.feature_name_])[:, 1]
        res["pred_target"] = mdl.predict(feat[mdl.feature_name_])
        return res

    @staticmethod
    def find_nearest(array, value):
        """
        Find the index of the value in the array nearest of the parameter value
        """
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    @staticmethod
    def points_to_plot(fpr, tpr, thresholds, threshold_to_find):
        """
        Find the closest values from the thresholds_to_find in thresholds and plot the points with the
        x value of FPR and y value of TPR
        It provides us to see by threshold the point in the AUC curve
        """
        fpr_to_plot = []
        tpr_to_plot = []
        thresholds_to_plot = []
        for element in threshold_to_find:
            index = function_utils.find_nearest(thresholds, element)
            fpr_to_plot.append(fpr[index])
            tpr_to_plot.append(tpr[index])
            thresholds_to_plot.append(thresholds[index])
        return fpr_to_plot, tpr_to_plot, thresholds_to_plot

    @staticmethod
    def print_roc_auc(listdict, title):
        plt.figure()
        lw = 2
        plt.plot([0, 1], [0, 1], color=plt.cm.tab10.colors[0], lw=lw, linestyle="--")

        for n, i in enumerate(listdict):
            fpr, tpr, thresholds = roc_curve(i["prediction"]["target"], i["prediction"]["proba"])
            fpr_to_plot, tpr_to_plot, thresholds_to_plot = function_utils.points_to_plot(
                fpr, tpr, thresholds, threshold_to_find=[0.25, 0.5, 0.7]
            )
            plt.scatter(fpr_to_plot, tpr_to_plot, color=plt.cm.tab10.colors[n + 1])
            for j in range(len(thresholds_to_plot)):
                plt.annotate(
                    round(thresholds_to_plot[j], 2),
                    (fpr_to_plot[j] + 0.02, tpr_to_plot[j] - 0.02),
                    color=plt.cm.tab10.colors[n + 1],
                )

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
        return plt, roc_auc

    @staticmethod
    def print_feature_importance(mdl, title="Feature importances"):
        """
        Print the features importances sorted from more important to less important
        """
        importances = pd.Series(mdl.feature_importances_, index=mdl.feature_name_)
        importances = importances.sort_values(ascending=False)

        fig, ax = plt.subplots()
        importances.plot.bar(ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        return plt

    @staticmethod
    def print_confusion_matrix(prediction, title, threshold=0.5):
        prediction["binarized_boolean"] = prediction["proba"] >= threshold
        cm = confusion_matrix(prediction["target"], prediction["binarized_boolean"])
        _ = ConfusionMatrixDisplay(cm).plot()
        plt.title(f"{title} - {threshold=}")
        return plt, cm

    @staticmethod
    def print_cross_validation(mdl, feat, target, cv, title):
        scores = cross_val_score(mdl, feat, target, cv=cv, scoring="roc_auc")
        print(scores)
        m_scores = round(mean(scores), 3)
        std_scores = round(std(scores), 3)
        print(title + " Accuracy: %.3f (%.3f)" % (m_scores, std_scores))
        return m_scores, std_scores

    @staticmethod
    def distribution(df):
        """
        Check the distribution, how many hotels or clients are needed to cover the 50% of the GS/GCC ?
        """

        p = (
            df.loc[
                df["FF_ONE_OM_SUM_L13S"] > 0.005,
                ["PK", "FF_ONE_OM_SUM_L13S"],
            ]
            .sort_values(by=["FF_ONE_OM_SUM_L13S"], ascending=False)
            .reset_index(drop=True)
        )

        p["CUM_ONE_OM_SUM_L13S"] = p["FF_ONE_OM_SUM_L13S"].cumsum()
        p["%_cum"] = p["CUM_ONE_OM_SUM_L13S"] / sum(p["FF_ONE_OM_SUM_L13S"])
        plt.figure()
        n = 50000
        p["%_cum"].head(n).plot(title="TOP: " + str(n) + "  out of total of: " + str(p.shape[0]))
        return plt

from pandas import DataFrame
import pandas as pd
from datetime import date
from project.functions import BaseFunctions, ApplyFunctions, apply_functions
from dateutil.relativedelta import relativedelta

"""
a) max_date = today
b) Simulate today is: break_target
c) Calculate the target, if the client is churn or not, with the data(*) from break_target to max_date.
 (*) these data are only used for that, by clause b) this should not exist.
"""


class TransformData:

    # at the moment, only permit group by months
    def __init__(self, df: DataFrame, max_date: date, months_break_target=1, features_definition_months=[]):
        self.df = df
        self.max_date = max_date
        self.break_target = max_date - relativedelta(months=months_break_target)
        self.features_definition_months = features_definition_months

        self.frequency_tag = "M"

    def transform_data(self):

        # calculate target
        df_target = self.calculate_target()

        # calculate features
        df_features = self.calculate_features()
        df_all = pd.concat([df_features, df_target], axis=1, join="inner")
        df_all["PK"] = df_all.index

        # return df with target and features
        return df_all

    def calculate_target(self):
        targets_row = self.df.loc[(self.break_target < self.df["DATE"]) & (self.df["DATE"] <= self.max_date), :]
        # rn -> gross sales
        targets_agg = (
            targets_row.groupby("PK").agg({"GROSS_SALES": "sum"}).rename(columns={"GROSS_SALES": "GROSS_SALES_TARGET"})
        )
        targets_agg["CHURN"] = targets_agg["GROSS_SALES_TARGET"].apply(lambda x: True if x < 0.005 else False)
        return targets_agg

    def calculate_features(self):

        features_agg = []
        for freq_months, metrics in self.features_definition_months.items():
            bottom_date = self.break_target - relativedelta(months=freq_months)
            features_row = self.df.loc[(bottom_date < self.df["DATE"]) & (self.df["DATE"] <= self.break_target), :]
            features_agg.append(self.execute_features(features_row, metrics, f"L{freq_months}{self.frequency_tag}"))

        features_agg = pd.concat(features_agg, axis=1, join="inner")
        return features_agg

    def execute_features(self, df_features, features_list, tag):
        # columns_rename = {}
        f_columns = {}
        f_isnull_date = {}
        f_apply = {}
        df_features_final = df_features.copy()
        for x in features_list:
            # columns_rename[x[0].upper()] = f"{x[0]}_{tag}".upper()
            list_func = x[1] if isinstance(x[1], list) else [x[1]]

            for func in list_func:
                # df.rename(columns_rename={x[0]: f"{x[0]}_{func.value}_{tag}".upper()})
                if func != BaseFunctions.ISNULL:

                    df_features_final[f"{x[0]}_{func.value}_{tag}".upper()] = df_features_final.loc[:, x[0].upper()]

                    if BaseFunctions.ISNULL in list_func:
                        f_isnull_date[f"{x[0]}_{func.value}_{tag}".upper()] = func
                    elif isinstance(func, ApplyFunctions):
                        f_apply[f"{x[0]}_{func.value}_{tag}".upper()] = func
                    else:
                        f_columns[f"{x[0]}_{func.value}_{tag}".upper()] = func.value

        df_basefunctions = df_features_final.groupby("PK").agg(f_columns)

        for col in f_isnull_date:
            function = f_isnull_date[col]
            if isinstance(function, BaseFunctions):
                df_ifnull = (
                    df_features_final.loc[~df_features_final[col].isna(), ["PK", col]]
                    .groupby("PK")
                    .agg({col: function.value})
                )
                df_basefunctions = df_basefunctions.join(df_ifnull, on="PK")
            elif isinstance(function, ApplyFunctions):
                result = apply_functions(df_features_final, col, function, context={"date": self.break_target})
                df_basefunctions = df_basefunctions.join(result, on="PK")

        for col in f_apply:
            function = f_isnull_date[col]
            result = apply_functions(df_features_final, col, function, context={"date": self.break_target})
            df_basefunctions = df_basefunctions.join(result, on="PK")

        return df_basefunctions

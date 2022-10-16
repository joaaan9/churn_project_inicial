from pandas import DataFrame
import pandas as pd
from functions import BaseFunctions, ApplyFunctions, apply_functions


class TransformData:
    def __init__(
        self, df: DataFrame, break_date: int, model="HOTEL", features_definition_months=[], target_combinations=[]
    ):
        self.df = df
        self.break_target = break_date
        self.date_time = df[df["TIME_SLOT"] == break_date]["DATE_TIME_SLOT"].unique()
        self.features_definition_months = features_definition_months
        self.target_combinations = target_combinations
        self.frequency_tag = "S"
        if model == "HOTEL":
            self.target_metric = "GROSS_COMMERCIAL_COST"
        else:
            self.target_metric = "GROSS_SALES"
        # The model can be HOTEL or CLIENT

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

        targets = []
        for i in self.target_combinations:
            targets_row = self.df.loc[(self.df["TIME_SLOT"] <= i["F_SLOT"]) & (self.df["TIME_SLOT"] >= i["L_SLOT"]), :]
            targets_agg = (
                targets_row.groupby("PK")
                .agg({self.target_metric: "sum"})
                .rename(columns={self.target_metric: f"{self.target_metric}_TARGET_PH_{i['PH']}_CP_{i['CP']}"})
            )
            targets_agg[f"CHURN_PH_{i['PH']}_CP_{i['CP']}S"] = targets_agg[
                f"{self.target_metric}_TARGET_PH_{i['PH']}_CP_{i['CP']}"
            ].apply(lambda x: True if x < 0.005 else False)
            targets.append(targets_agg)

        targets = pd.concat(targets, axis=1, join="inner")
        return targets

    def calculate_features(self):
        features_agg = []
        for freq_months_agg, metrics in self.features_definition_months.items():
            freq_months = freq_months_agg[0]
            agg = freq_months_agg[1]
            if agg == "agg":
                bottom_date = self.break_target + freq_months
                features_row = self.df.loc[
                    (self.break_target <= self.df["TIME_SLOT"]) & (self.df["TIME_SLOT"] < bottom_date), :
                ]
                features_agg.append(self.execute_features(features_row, metrics, f"L{freq_months}{self.frequency_tag}"))
            else:
                bottom_date = self.break_target + freq_months - 1
                features_row = self.df.loc[(self.df["TIME_SLOT"] == bottom_date), :]
                features_agg.append(self.execute_features(features_row, metrics, f"L{freq_months}{self.frequency_tag}O"))

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
                if func != BaseFunctions.ISNULL:

                    df_features_final[f"FF_{x[0]}_{func.value}_{tag}".upper()] = df_features_final.loc[:, x[0].upper()]

                    if BaseFunctions.ISNULL in list_func:
                        f_isnull_date[f"FF_{x[0]}_{func.value}_{tag}".upper()] = func
                    elif isinstance(func, ApplyFunctions):
                        f_apply[f"FF_{x[0]}_{func.value}_{tag}".upper()] = func
                    else:
                        f_columns[f"FF_{x[0]}_{func.value}_{tag}".upper()] = func.value

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
                result = apply_functions(df_features_final, col, function, context={"date": self.date_time})
                df_basefunctions = df_basefunctions.join(result, on="PK")

        for col in f_apply:
            function = f_isnull_date[col]
            result = apply_functions(df_features_final, col, function, context={"date": self.date_time})
            df_basefunctions = df_basefunctions.join(result, on="PK")

        return df_basefunctions

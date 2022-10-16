import time
import project.util.snowflake as sf
from project.functions import BaseFunctions, ApplyFunctions
from project.functions_utils import function_utils
from project.transform_data import TransformData

# If agg = "agg" then all slots from months_agg until break are aggregated
# If agg = "only" then only is consider the slot of months_agg
agg = "agg"

# If model it's HOTEL is used the gross commercial cost as a target
# Is model it's CLIENT is used the gross sales as a target
# Also the pickle and the query of DBT change
MODEL = "HOTEL"

months_agg = [1, 2, 3, 4, 13]

features_definition_m = [
    (("rn", BaseFunctions.SUM), months_agg, agg),
    (("gross_sales", BaseFunctions.SUM), months_agg, agg),
    (("one_om", BaseFunctions.SUM), months_agg, agg),
    (("gross_commercial_cost", BaseFunctions.SUM), months_agg, agg),
    (("pax", BaseFunctions.SUM), months_agg, agg),
    (("booking_dates", BaseFunctions.SUM), months_agg, agg),
    (("markup", BaseFunctions.AVG), months_agg, agg),
    (("markup", BaseFunctions.AVG), months_agg, "only"),
    (("cpt_p", BaseFunctions.AVG), months_agg, agg),
    (("cpt_all", BaseFunctions.AVG), months_agg, agg),
    (("cpt_all", BaseFunctions.AVG), months_agg, "only"),
    (("searches", BaseFunctions.SUM), months_agg, agg),
    (("searches", BaseFunctions.SUM), months_agg, "only"),
    (("valuations", BaseFunctions.SUM), months_agg, agg),
    (("valuations", BaseFunctions.SUM), months_agg, "only"),
    (("searches_1_hotel", BaseFunctions.SUM), months_agg, agg),
    (("searches_1_hotel", BaseFunctions.SUM), months_agg, "only"),
    (("times_returned_1_hotel", BaseFunctions.SUM), months_agg, agg),
    (("times_returned_1_hotel", BaseFunctions.SUM), months_agg, "only"),
    (("maxirooms_events", BaseFunctions.SUM), months_agg, agg),
    (("min_booking_date", [BaseFunctions.ISNULL, ApplyFunctions.DIFF_DATE_MIN]), months_agg, agg),
    (("max_booking_date", [BaseFunctions.ISNULL, ApplyFunctions.DIFF_DATE_MAX]), months_agg, agg),
    (("hotel_category_group", BaseFunctions.MAX), [1], agg),
    (("ratemix", BaseFunctions.MAX), months_agg, agg),
    (("ratemix", BaseFunctions.MAX), months_agg, "only"),
    (("has_contracts_to_future", BaseFunctions.MAX), [1], agg),
    (("has_some_gnd", BaseFunctions.MAX), [1], agg),
    (("has_gnd", BaseFunctions.MAX), [1], agg),
    (("segmentation", BaseFunctions.MAX), [1], agg),
    (("hotel_destination_segment", BaseFunctions.MAX), [1], agg),
    (("region", BaseFunctions.MAX), [1], agg),
    (("hotel_total_rooms", BaseFunctions.MAX), [1], agg),
    (("days_to_date_from_gnd", BaseFunctions.MAX), [1], agg),
    (("days_to_date_to_gnd", BaseFunctions.MAX), [1], agg),
    (("days_to_max_arrival_date", BaseFunctions.MAX), [1], agg),
    (("gross_commercial_cost_next_slot", BaseFunctions.SUM), [1], agg),
    (("churn_2m", BaseFunctions.SUM), [6, 13], agg),
    (("churn_3m", BaseFunctions.SUM), [6, 13], agg),
    (("churn_4m", BaseFunctions.SUM), [6, 13], agg),
    (("churn_5m", BaseFunctions.SUM), [6, 13], agg),
    (("churn_more_5m", BaseFunctions.SUM), [6, 13], agg),
]


if __name__ == "__main__":

    target_combinations = function_utils.get_all_targets_combinations()

    SCENARIO_DATE = 4
    sn = sf.Snowflake()
    df = function_utils.load_data(sn, in_pickle=True, model=MODEL)
    print("Data Loaded!")

    inicio = time.time()
    result_today = TransformData(
        df,
        break_date=SCENARIO_DATE,
        model=MODEL,
        features_definition_months=function_utils.get_list_features(features_definition_m),
        target_combinations=target_combinations,
    ).transform_data()
    result_today["SCENARIO_DATE"] = SCENARIO_DATE
    # Checking the data
    print("Done scenario (today): " + str(SCENARIO_DATE))
    print(result_today.agg({"GROSS_COMMERCIAL_COST_TARGET_PH_1_CP_1": "sum", "FF_GROSS_COMMERCIAL_COST_SUM_L1S": "sum"}))
    function_utils.save_data(result_today, "df_transform_today_hotel_churn", model=MODEL)

    result_scenarios = function_utils.load_data_more_scenarios(
        df,
        features=function_utils.get_list_features(features_definition_m),
        start_time=(SCENARIO_DATE + 1),
        end_time=(SCENARIO_DATE + 5),
        len=SCENARIO_DATE,
        model=MODEL,
        target_combinations=target_combinations,
    )
    function_utils.save_data(result_scenarios, "df_transform_scenarios_hotel_churn", model=MODEL)

    fin = time.time()
    print(fin - inicio)

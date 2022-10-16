import os
from collections import defaultdict
import time

import project.util.snowflake as sf
import pandas as pd

from datetime import date

from project.functions import BaseFunctions, ApplyFunctions
from functools import reduce

from project.transform_data import TransformData

"""
features_definition_months=[
    (("RN", BaseFunctions.SUM), [1,2,3,12]), #RN_L2M, RN_L2M, RN_LAG_2M , RN_LAG_3M, (1,3)
    (("RN", if_0), [2,12]),
    (("SEARCHES_TOTAL", BaseFunctions.SUM),[2,12]),
    (("BOOKINGS_WITH_CASES_OPS", BaseFunctions.SUM),[2,12])
]
features_definition_condition=[
    (("RN", BaseFunctions.DATE_MIN_MAX), [12]),
]
"""

months_agg = [1, 12]

features_definition_months = [
    # Trading
    (("rn", BaseFunctions.SUM), months_agg),
    (("gross_sales", BaseFunctions.SUM), months_agg),
    (("booking_dates", BaseFunctions.SUM), months_agg),
    (("min_booking_date", [BaseFunctions.ISNULL, ApplyFunctions.DIFF_DATE_MIN]), months_agg),
    (("max_booking_date", [BaseFunctions.ISNULL, ApplyFunctions.DIFF_DATE_MAX]), months_agg),
    # To define churn
    # (("rn", if_0), months_agg),
    # Searches (pending some in python)
    (("searches_total", BaseFunctions.SUM), months_agg),
    (("valuations_total", BaseFunctions.SUM), months_agg),
    (("bookings_total", BaseFunctions.SUM), months_agg),
    (("bookingcancellation", BaseFunctions.SUM), months_agg),
    (("searches_failed", BaseFunctions.SUM), months_agg),
    (("valuations_failed", BaseFunctions.SUM), months_agg),
    (("bookings_failed", BaseFunctions.SUM), months_agg),
    # Cases
    (("bookings_with_cases_ops", BaseFunctions.SUM), months_agg),
    (("bookings_with_cases_sales", BaseFunctions.SUM), months_agg),
    (("bookings_with_cases_finance", BaseFunctions.SUM), months_agg),
    (("bookout", BaseFunctions.SUM), months_agg),
    (("cases_finance_reason_fm", BaseFunctions.SUM), months_agg),
    (("cases_finance_status_solved", BaseFunctions.SUM), months_agg),
    (("cases_ops_status_solved", BaseFunctions.SUM), months_agg),
    (("cases_ops_status_noproceed", BaseFunctions.SUM), months_agg),
    (("cases_sales_status_solved", BaseFunctions.SUM), months_agg),
    # (("bookingcancellationrequest", BaseFunctions.SUM), months_agg),
    (("bookingcancellationandorwaiverrequest", BaseFunctions.SUM), months_agg),
    (("bookingnotfoundorbookingdiscrepancies", BaseFunctions.SUM), months_agg),
    (("bookingpaidatdestination", BaseFunctions.SUM), months_agg),
    (("bookingstatusandbookingreconfirmation", BaseFunctions.SUM), months_agg),
    (("breachofcontract", BaseFunctions.SUM), months_agg),
    (("businessmodelcommission", BaseFunctions.SUM), months_agg),
    (("cancellation", BaseFunctions.SUM), months_agg),
    (("clientnoshow", BaseFunctions.SUM), months_agg),
    (("commercialactionsgestures", BaseFunctions.SUM), months_agg),
    (("commercialrequestsupport", BaseFunctions.SUM), months_agg),
    (("compensationissues", BaseFunctions.SUM), months_agg),
    (("connectivity", BaseFunctions.SUM), months_agg),
    (("contenterror", BaseFunctions.SUM), months_agg),
    (("contentweberrororsystemfailure", BaseFunctions.SUM), months_agg),
    (("contracterror", BaseFunctions.SUM), months_agg),
    (("contractunification", BaseFunctions.SUM), months_agg),
    (("creation", BaseFunctions.SUM), months_agg),
    (("creditcardfraudccf", BaseFunctions.SUM), months_agg),
    (("creditcontrol", BaseFunctions.SUM), months_agg),
    (("dissatisfaction", BaseFunctions.SUM), months_agg),
    (("doesnotcorrespondtoagency", BaseFunctions.SUM), months_agg),
    (("doublechargeserroneouscharges", BaseFunctions.SUM), months_agg),
    (("duplicatedinvoice", BaseFunctions.SUM), months_agg),
    (("feesnegotiationrequired", BaseFunctions.SUM), months_agg),
    (("feesnegotiationrequested", BaseFunctions.SUM), months_agg),
    (("fraudbookings", BaseFunctions.SUM), months_agg),
    (("fraudulentagencyfag", BaseFunctions.SUM), months_agg),
    (("fraudulentbooking", BaseFunctions.SUM), months_agg),
    (("hotelincidentduringstay", BaseFunctions.SUM), months_agg),
    (("hotelchangeterminaltoterminaltransfer", BaseFunctions.SUM), months_agg),
    (("hotelclosed", BaseFunctions.SUM), months_agg),
    (("hotelbedscompanybrandinfo", BaseFunctions.SUM), months_agg),
    (("hotelbedshotelroomtypedifferences", BaseFunctions.SUM), months_agg),
    (("inflightservice", BaseFunctions.SUM), months_agg),
    (("inconveniencesduringrentacarpickupreturn", BaseFunctions.SUM), months_agg),
    (("inconveniencesduringticketpickup", BaseFunctions.SUM), months_agg),
    (("inconveniencesduringticketutilization", BaseFunctions.SUM), months_agg),
    (("massiveoverselling", BaseFunctions.SUM), months_agg),
    (("noshow", BaseFunctions.SUM), months_agg),
    (("overbooking", BaseFunctions.SUM), months_agg),
    (("paymentissue", BaseFunctions.SUM), months_agg),
    (("prepayment", BaseFunctions.SUM), months_agg),
    (("pricematch", BaseFunctions.SUM), months_agg),
    (("ratediscrepancy", BaseFunctions.SUM), months_agg),
    (("ratedisputeorreservationcontracterror", BaseFunctions.SUM), months_agg),
    (("ratesandfees", BaseFunctions.SUM), months_agg),
    (("requestvalidcreditcard", BaseFunctions.SUM), months_agg),
    (("resendvoucher", BaseFunctions.SUM), months_agg),
    (("reservationerrorornotfound", BaseFunctions.SUM), months_agg),
    (("returnoperationproblem", BaseFunctions.SUM), months_agg),
    (("salessupportwebcredentials", BaseFunctions.SUM), months_agg),
    (("servicedescriptioninnacurate", BaseFunctions.SUM), months_agg),
    (("servicenotprovided", BaseFunctions.SUM), months_agg),
    (("stopsales", BaseFunctions.SUM), months_agg),
    (("systemissue", BaseFunctions.SUM), months_agg),
    (("weberrorsystemfailure", BaseFunctions.SUM), months_agg),
    (("groups", BaseFunctions.SUM), months_agg),
    (("extras", BaseFunctions.SUM), months_agg),
    (("hotel_contract", BaseFunctions.SUM), months_agg),
    (("accomocation", BaseFunctions.SUM), months_agg),
    (("compensation", BaseFunctions.SUM), months_agg),
    (("transfer", BaseFunctions.SUM), months_agg),
    (("circuits", BaseFunctions.SUM), months_agg),
    (("totality_service", BaseFunctions.SUM), months_agg),
    (("excursions", BaseFunctions.SUM), months_agg),
    (("specialist_tours", BaseFunctions.SUM), months_agg),
    # Overrides
    (("HAS_OVERRIDE", BaseFunctions.MAX), months_agg),
    (("AMOUNT_OVERRIDE_EUR", BaseFunctions.SUM), months_agg),
    (("OVERRIDE_TRIGGERED", BaseFunctions.MAX), months_agg),
    # markup
    (("markup", BaseFunctions.AVG), months_agg),
    (("channel_markup", BaseFunctions.AVG), months_agg),
    # prepayment or credit
    (("payment_credit", BaseFunctions.MAX), months_agg),
    (("payment_prepayment", BaseFunctions.MAX), months_agg),
    # res_cases
    (("res_ghost_bookings", BaseFunctions.SUM), months_agg),
    (("res_breach_of_contract", BaseFunctions.SUM), months_agg),
    (("res_building_works", BaseFunctions.SUM), months_agg),
    (("res_connectivity", BaseFunctions.SUM), months_agg),
    (("res_force_majeure", BaseFunctions.SUM), months_agg),
    (("res_hotel_closure", BaseFunctions.SUM), months_agg),
    (("res_overbooking_overselling", BaseFunctions.SUM), months_agg),
    (("res_payment_issue", BaseFunctions.SUM), months_agg),
    (("res_system_issue", BaseFunctions.SUM), months_agg),
    (("res_contract_unification", BaseFunctions.SUM), months_agg),
    (("res_refundable", BaseFunctions.SUM), months_agg),
    (("res_non_refundable", BaseFunctions.SUM), months_agg),
    (("res_agent_mistake", BaseFunctions.SUM), months_agg),
    (("res_booking_maximization", BaseFunctions.SUM), months_agg),
    (("res_cancellation_fees", BaseFunctions.SUM), months_agg),
    (("res_bankruptcy", BaseFunctions.SUM), months_agg),
    (("res_customer_complaint", BaseFunctions.SUM), months_agg),
    (("res_commercial_gesture", BaseFunctions.SUM), months_agg),
    (("res_contract_error", BaseFunctions.SUM), months_agg),
    (("res_debt_writeoff", BaseFunctions.SUM), months_agg),
    (("res_denied_bookings", BaseFunctions.SUM), months_agg),
    (("res_fraudulent_bookings", BaseFunctions.SUM), months_agg),
    (("res_grace_period", BaseFunctions.SUM), months_agg),
    (("res_guarantees", BaseFunctions.SUM), months_agg),
    (("res_mapping_error", BaseFunctions.SUM), months_agg),
    (("res_no_show", BaseFunctions.SUM), months_agg),
    (("res_pink_booking", BaseFunctions.SUM), months_agg),
    (("res_stop_sales", BaseFunctions.SUM), months_agg),
    (("res_btb", BaseFunctions.SUM), months_agg),
    (("res_vacation_rental", BaseFunctions.SUM), months_agg),
    (("res_requested_by_client", BaseFunctions.SUM), months_agg),
    (("res_bco", BaseFunctions.SUM), months_agg),
    (("res_canrec", BaseFunctions.SUM), months_agg),
    (("res_change_of_currency", BaseFunctions.SUM), months_agg),
    (("res_change_of_taxes", BaseFunctions.SUM), months_agg),
    (("res_b2b_configuration", BaseFunctions.SUM), months_agg),
    (("res_uncatalogued", BaseFunctions.SUM), months_agg),
    (("res_cancellation", BaseFunctions.SUM), months_agg),
    (("res_tax_adjustment", BaseFunctions.SUM), months_agg),
    (("res_massive_overselling", BaseFunctions.SUM), months_agg),
    (("res_other_modifications", BaseFunctions.SUM), months_agg),
    (("res_system_decommissioning", BaseFunctions.SUM), months_agg),
    (("res_crc", BaseFunctions.SUM), months_agg),
    (("res_fraud", BaseFunctions.SUM), months_agg),
    (("res_dsweb_out_of_scope", BaseFunctions.SUM), months_agg),
    (("res_old_bookings", BaseFunctions.SUM), months_agg),
    (("res_amendment_by_client", BaseFunctions.SUM), months_agg),
    (("res_ds_out_of_scope", BaseFunctions.SUM), months_agg),
    (("res_legacy_system", BaseFunctions.SUM), months_agg),
    # calls
    (("calls_num_on_hold_cases", BaseFunctions.SUM), months_agg),
    (("calls_num_closed_cases", BaseFunctions.SUM), months_agg),
    (("calls_num_other_cases", BaseFunctions.SUM), months_agg),
    (("calls_total_num_cases", BaseFunctions.SUM), months_agg),
    (("calls_avg_case_aging", BaseFunctions.AVG), months_agg),
    (("calls_avg_fcr", BaseFunctions.AVG), months_agg),
    (("calls_avg_answer_time", BaseFunctions.SUM), months_agg),
    (("calls_sum_calls", BaseFunctions.SUM), months_agg),
    (("calls_sum_attended", BaseFunctions.SUM), months_agg),
    (("calls_avg_sla", BaseFunctions.AVG), months_agg),
    (("calls_sum_avrg_aht", BaseFunctions.SUM), months_agg),
]


def load_data(sn):
    # cur = sn.cnx.cursor()
    # cur.execute(" SELECT * FROM HBG_DATASCIENCE.SANDBOX_ANALYTICS.FCT_CHURN_MODELLING_DATA_AGG ")
    # df = cur.fetch_pandas_all()

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))), "output/df_raw.pickle")
    # path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "output/df_raw.pickle")
    # df.to_pickle(path)
    # Or, load existing data
    df = pd.read_pickle(path)
    return df


def save_data(df, name):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath("__file__"))), f"output/{name}.pickle")
    # path = os.path.join(os.path.dirname(os.path.abspath("__file__")), f"output/{name}.pickle")

    df.to_pickle(path)


def load_data_more_scenarios(df, features, start_date="2021-01", end_date="2022-02"):

    df_months = pd.date_range(start_date, end_date, freq="MS").sort_values(ascending=False)
    scenarios = []

    for m in df_months:
        result = TransformData(df, max_date=m.date(), features_definition_months=features).transform_data()
        result["SCENARIO_DATE"] = m
        # save_data(result, f'df_transform_{m.date()}')
        scenarios.append(result)

    df_all = pd.concat(scenarios, ignore_index=True)

    return df_all


def get_list_features(features):
    # Metric, operacion, meses
    # RN, sum, [1, 2, 18]
    # MARKUP, avg, [2, 5]
    list_features = list(map(lambda val: [(x, val[0]) for x in val[1]], features))
    list_features = reduce(lambda a, b: a + b, list_features)
    list_features = reduce(lambda grp, val: grp[val[0]].append(val[1]) or grp, list_features, defaultdict(list))
    return list_features


if __name__ == "__main__":

    """
    path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "output/df_transform_scenarios.pickle")
    df1 = pd.read_pickle(path)

    path = os.path.join(os.path.dirname(os.path.abspath("__file__")), "output/df_transform_today.pickle")
    df2 = pd.read_pickle(path)
    """

    MAX_DATE = date(2022, 3, 1)
    # get_list_features(features_definition_months)
    sn = sf.Snowflake()
    df = load_data(sn)

    # Remove all wholesale clients
    # df = df[[i if i is not None else True for i in df.CUSTOMER_RL_BL_RETAIL]]

    inicio = time.time()
    result1 = TransformData(
        df,
        max_date=MAX_DATE,
        months_break_target=1,
        features_definition_months=get_list_features(features_definition_months),
    ).transform_data()
    result1["SCENARIO_DATE"] = MAX_DATE
    save_data(result1, "df_transform_today")

    result = load_data_more_scenarios(df, features=get_list_features(features_definition_months))
    save_data(result, "df_transform_scenarios")

    fin = time.time()

    print(fin - inicio)

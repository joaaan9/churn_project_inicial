import time

import os
import mlflow

import util.snowflake as sf
from functions import BaseFunctions, ApplyFunctions
from transform_data import TransformData
from functions_utils import function_utils

# If agg = "agg" then all slots from months_agg until break are aggregated
# If agg = "only" then only is consider the slot of months_agg
agg = "agg"

# If model it's HOTEL is used the gross commercial cost as a target
# Is model it's CLIENT is used the gross sales as a target
# Also the pickle and the query of DBT change
MODEL = "CLIENT"

months_agg = [1, 2, 3, 4, 13]

features_definition_m = [
    # Trading
    (("rn", BaseFunctions.SUM), months_agg, agg),
    (("gross_sales", BaseFunctions.SUM), months_agg, agg),
    (("one_om", BaseFunctions.SUM), months_agg, agg),
    (("booking_dates", BaseFunctions.SUM), months_agg, agg),
    (("min_booking_date", [BaseFunctions.ISNULL, ApplyFunctions.DIFF_DATE_MIN]), months_agg, agg),
    (("max_booking_date", [BaseFunctions.ISNULL, ApplyFunctions.DIFF_DATE_MAX]), months_agg, agg),
    (("searches_total", BaseFunctions.SUM), months_agg, agg),
    (("valuations_total", BaseFunctions.SUM), months_agg, agg),
    (("bookings_total", BaseFunctions.SUM), months_agg, agg),
    (("bookingcancellation", BaseFunctions.SUM), months_agg, agg),
    (("searches_failed", BaseFunctions.SUM), months_agg, agg),
    (("valuations_failed", BaseFunctions.SUM), months_agg, agg),
    (("bookings_failed", BaseFunctions.SUM), months_agg, agg),
    # Cases
    (("bookings_with_cases_ops", BaseFunctions.SUM), months_agg, agg),
    (("bookings_with_cases_sales", BaseFunctions.SUM), months_agg, agg),
    (("bookings_with_cases_finance", BaseFunctions.SUM), months_agg, agg),
    (("bookout", BaseFunctions.SUM), months_agg, agg),
    (("cases_finance_reason_fm", BaseFunctions.SUM), months_agg, agg),
    (("cases_finance_status_solved", BaseFunctions.SUM), months_agg, agg),
    (("cases_ops_status_solved", BaseFunctions.SUM), months_agg, agg),
    (("cases_ops_status_noproceed", BaseFunctions.SUM), months_agg, agg),
    (("cases_sales_status_solved", BaseFunctions.SUM), months_agg, agg),
    # (("bookingcancellationrequest", BaseFunctions.SUM), months_agg, agg),
    (("bookingcancellationandorwaiverrequest", BaseFunctions.SUM), months_agg, agg),
    (("bookingnotfoundorbookingdiscrepancies", BaseFunctions.SUM), months_agg, agg),
    (("bookingpaidatdestination", BaseFunctions.SUM), months_agg, agg),
    (("bookingstatusandbookingreconfirmation", BaseFunctions.SUM), months_agg, agg),
    (("breachofcontract", BaseFunctions.SUM), months_agg, agg),
    (("businessmodelcommission", BaseFunctions.SUM), months_agg, agg),
    (("cancellation", BaseFunctions.SUM), months_agg, agg),
    (("clientnoshow", BaseFunctions.SUM), months_agg, agg),
    (("commercialactionsgestures", BaseFunctions.SUM), months_agg, agg),
    (("commercialrequestsupport", BaseFunctions.SUM), months_agg, agg),
    (("compensationissues", BaseFunctions.SUM), months_agg, agg),
    (("connectivity", BaseFunctions.SUM), months_agg, agg),
    (("contenterror", BaseFunctions.SUM), months_agg, agg),
    (("contentweberrororsystemfailure", BaseFunctions.SUM), months_agg, agg),
    (("contracterror", BaseFunctions.SUM), months_agg, agg),
    (("contractunification", BaseFunctions.SUM), months_agg, agg),
    (("creation", BaseFunctions.SUM), months_agg, agg),
    (("creditcardfraudccf", BaseFunctions.SUM), months_agg, agg),
    (("creditcontrol", BaseFunctions.SUM), months_agg, agg),
    (("dissatisfaction", BaseFunctions.SUM), months_agg, agg),
    (("doesnotcorrespondtoagency", BaseFunctions.SUM), months_agg, agg),
    (("doublechargeserroneouscharges", BaseFunctions.SUM), months_agg, agg),
    (("duplicatedinvoice", BaseFunctions.SUM), months_agg, agg),
    (("feesnegotiationrequired", BaseFunctions.SUM), months_agg, agg),
    (("feesnegotiationrequested", BaseFunctions.SUM), months_agg, agg),
    (("fraudbookings", BaseFunctions.SUM), months_agg, agg),
    (("fraudulentagencyfag", BaseFunctions.SUM), months_agg, agg),
    (("fraudulentbooking", BaseFunctions.SUM), months_agg, agg),
    (("hotelincidentduringstay", BaseFunctions.SUM), months_agg, agg),
    (("hotelchangeterminaltoterminaltransfer", BaseFunctions.SUM), months_agg, agg),
    (("hotelclosed", BaseFunctions.SUM), months_agg, agg),
    (("hotelbedscompanybrandinfo", BaseFunctions.SUM), months_agg, agg),
    (("hotelbedshotelroomtypedifferences", BaseFunctions.SUM), months_agg, agg),
    (("inflightservice", BaseFunctions.SUM), months_agg, agg),
    (("inconveniencesduringrentacarpickupreturn", BaseFunctions.SUM), months_agg, agg),
    (("inconveniencesduringticketpickup", BaseFunctions.SUM), months_agg, agg),
    (("inconveniencesduringticketutilization", BaseFunctions.SUM), months_agg, agg),
    (("massiveoverselling", BaseFunctions.SUM), months_agg, agg),
    (("noshow", BaseFunctions.SUM), months_agg, agg),
    (("overbooking", BaseFunctions.SUM), months_agg, agg),
    (("paymentissue", BaseFunctions.SUM), months_agg, agg),
    (("prepayment", BaseFunctions.SUM), months_agg, agg),
    (("pricematch", BaseFunctions.SUM), months_agg, agg),
    (("ratediscrepancy", BaseFunctions.SUM), months_agg, agg),
    (("ratedisputeorreservationcontracterror", BaseFunctions.SUM), months_agg, agg),
    (("ratesandfees", BaseFunctions.SUM), months_agg, agg),
    (("requestvalidcreditcard", BaseFunctions.SUM), months_agg, agg),
    (("resendvoucher", BaseFunctions.SUM), months_agg, agg),
    (("reservationerrorornotfound", BaseFunctions.SUM), months_agg, agg),
    (("returnoperationproblem", BaseFunctions.SUM), months_agg, agg),
    (("salessupportwebcredentials", BaseFunctions.SUM), months_agg, agg),
    (("servicedescriptioninnacurate", BaseFunctions.SUM), months_agg, agg),
    (("servicenotprovided", BaseFunctions.SUM), months_agg, agg),
    (("stopsales", BaseFunctions.SUM), months_agg, agg),
    (("systemissue", BaseFunctions.SUM), months_agg, agg),
    (("weberrorsystemfailure", BaseFunctions.SUM), months_agg, agg),
    (("groups", BaseFunctions.SUM), months_agg, agg),
    (("extras", BaseFunctions.SUM), months_agg, agg),
    (("hotel_contract", BaseFunctions.SUM), months_agg, agg),
    (("accomocation", BaseFunctions.SUM), months_agg, agg),
    (("compensation", BaseFunctions.SUM), months_agg, agg),
    (("transfer", BaseFunctions.SUM), months_agg, agg),
    (("circuits", BaseFunctions.SUM), months_agg, agg),
    (("totality_service", BaseFunctions.SUM), months_agg, agg),
    (("excursions", BaseFunctions.SUM), months_agg, agg),
    (("specialist_tours", BaseFunctions.SUM), months_agg, agg),
    # Overrides
    # (("HAS_OVERRIDE", BaseFunctions.MAX), months_agg, agg),
    (("AMOUNT_OVERRIDE_EUR", BaseFunctions.SUM), months_agg, agg),
    # (("OVERRIDE_TRIGGERED", BaseFunctions.MAX), months_agg, agg),
    # markup
    (("markup", BaseFunctions.AVG), months_agg, agg),
    (("channel_markup", BaseFunctions.AVG), months_agg, agg),
    # prepayment or credit
    # (("payment_credit", BaseFunctions.MAX), months_agg, agg),
    # (("payment_prepayment", BaseFunctions.MAX), months_agg, agg),
    # res_cases
    # (("res_ghost_bookings", BaseFunctions.SUM), months_agg, agg),
    # (("res_breach_of_contract", BaseFunctions.SUM), months_agg, agg),
    # (("res_building_works", BaseFunctions.SUM), months_agg, agg),
    # (("res_connectivity", BaseFunctions.SUM), months_agg, agg),
    # (("res_force_majeure", BaseFunctions.SUM), months_agg, agg),
    # (("res_hotel_closure", BaseFunctions.SUM), months_agg, agg),
    # (("res_overbooking_overselling", BaseFunctions.SUM), months_agg, agg),
    # (("res_payment_issue", BaseFunctions.SUM), months_agg, agg),
    # (("res_system_issue", BaseFunctions.SUM), months_agg, agg),
    # (("res_contract_unification", BaseFunctions.SUM), months_agg, agg),
    # (("res_refundable", BaseFunctions.SUM), months_agg, agg),
    # (("res_non_refundable", BaseFunctions.SUM), months_agg, agg),
    # (("res_agent_mistake", BaseFunctions.SUM), months_agg, agg),
    # (("res_booking_maximization", BaseFunctions.SUM), months_agg, agg),
    # (("res_cancellation_fees", BaseFunctions.SUM), months_agg, agg),
    # (("res_bankruptcy", BaseFunctions.SUM), months_agg, agg),
    # (("res_customer_complaint", BaseFunctions.SUM), months_agg, agg),
    # (("res_commercial_gesture", BaseFunctions.SUM), months_agg, agg),
    # (("res_contract_error", BaseFunctions.SUM), months_agg, agg),
    # (("res_debt_writeoff", BaseFunctions.SUM), months_agg, agg),
    # (("res_denied_bookings", BaseFunctions.SUM), months_agg, agg),
    # (("res_fraudulent_bookings", BaseFunctions.SUM), months_agg, agg),
    # (("res_grace_period", BaseFunctions.SUM), months_agg, agg),
    # (("res_guarantees", BaseFunctions.SUM), months_agg, agg),
    # (("res_mapping_error", BaseFunctions.SUM), months_agg, agg),
    # (("res_no_show", BaseFunctions.SUM), months_agg, agg),
    # (("res_pink_booking", BaseFunctions.SUM), months_agg, agg),
    # (("res_stop_sales", BaseFunctions.SUM), months_agg, agg),
    # (("res_btb", BaseFunctions.SUM), months_agg, agg),
    # (("res_vacation_rental", BaseFunctions.SUM), months_agg, agg),
    # (("res_requested_by_client", BaseFunctions.SUM), months_agg, agg),
    # (("res_bco", BaseFunctions.SUM), months_agg, agg),
    # (("res_canrec", BaseFunctions.SUM), months_agg, agg),
    # (("res_change_of_currency", BaseFunctions.SUM), months_agg, agg),
    # (("res_change_of_taxes", BaseFunctions.SUM), months_agg, agg),
    # (("res_b2b_configuration", BaseFunctions.SUM), months_agg, agg),
    # (("res_uncatalogued", BaseFunctions.SUM), months_agg, agg),
    # (("res_cancellation", BaseFunctions.SUM), months_agg, agg),
    # (("res_tax_adjustment", BaseFunctions.SUM), months_agg, agg),
    # (("res_massive_overselling", BaseFunctions.SUM), months_agg, agg),
    # (("res_other_modifications", BaseFunctions.SUM), months_agg, agg),
    # (("res_system_decommissioning", BaseFunctions.SUM), months_agg, agg),
    # (("res_crc", BaseFunctions.SUM), months_agg, agg),
    # (("res_fraud", BaseFunctions.SUM), months_agg, agg),
    # (("res_dsweb_out_of_scope", BaseFunctions.SUM), months_agg, agg),
    # (("res_old_bookings", BaseFunctions.SUM), months_agg, agg),
    # (("res_amendment_by_client", BaseFunctions.SUM), months_agg, agg),
    # (("res_ds_out_of_scope", BaseFunctions.SUM), months_agg, agg),
    # (("res_legacy_system", BaseFunctions.SUM), months_agg, agg),
    # calls
    (("calls_num_on_hold_cases", BaseFunctions.SUM), months_agg, agg),
    (("calls_num_closed_cases", BaseFunctions.SUM), months_agg, agg),
    (("calls_num_other_cases", BaseFunctions.SUM), months_agg, agg),
    (("calls_total_num_cases", BaseFunctions.SUM), months_agg, agg),
    (("calls_avg_case_aging", BaseFunctions.AVG), months_agg, agg),
    (("calls_avg_fcr", BaseFunctions.AVG), months_agg, agg),
    (("calls_avg_answer_time", BaseFunctions.SUM), months_agg, agg),
    (("calls_sum_calls", BaseFunctions.SUM), months_agg, agg),
    (("calls_sum_attended", BaseFunctions.SUM), months_agg, agg),
    (("calls_avg_sla", BaseFunctions.AVG), months_agg, agg),
    (("calls_sum_avrg_aht", BaseFunctions.SUM), months_agg, agg),
]

if __name__ == "__main__":
    mlflow.set_tracking_uri('http://localhost:5000')
    with mlflow.start_run() as mlrun:
        target_combinations = function_utils.get_all_targets_combinations()
        print("Processing data")

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
        print(result_today.agg({"GROSS_SALES_TARGET_PH_1_CP_1": "sum", "FF_GROSS_SALES_SUM_L1S": "sum"}))
        function_utils.save_data(result_today, "df_transform_today_client_churn", model=MODEL)

        result_scenarios = function_utils.load_data_more_scenarios(
            df,
            features=function_utils.get_list_features(features_definition_m),
            start_time=(SCENARIO_DATE + 1),
            end_time=(SCENARIO_DATE + 5),
            len=SCENARIO_DATE,
            model=MODEL,
            target_combinations=target_combinations,
        )
        function_utils.save_data(result_scenarios, "df_transform_scenarios_client_churn", model=MODEL)

        # log pickle files as mlflow artifacts
        # get paths to pickle files
        project_path = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
        path_raw_pickle = os.path.join(project_path, "output/df_raw_client_churn.pickle")
        path_today_pickle = os.path.join(project_path, "output/df_transform_today_client_churn.pickle")
        path_scenarios_pickle = os.path.join(project_path, "output/df_transform_scenarios_client_churn.pickle")
        # log as artifacts
        artifacts_directory = "df_pickles"
        mlflow.log_artifact(path_raw_pickle, artifacts_directory)
        artifacts_directory = artifacts_directory + "/transformed"
        mlflow.log_artifact(path_today_pickle, artifacts_directory)
        mlflow.log_artifact(path_scenarios_pickle, artifacts_directory)

        fin = time.time()

        print(fin - inicio)

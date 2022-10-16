import datetime as dt

import boruta
import lightgbm
import pandas as pd
import shap
import sklearn.metrics as sk_mtrc

# import sklearn.preprocessing as sk_prep
# import sklearn.ensemble as sk_ensm
import sklearn.model_selection as sk_ms

import project.util.snowflake as util_sf

connection = util_sf.Snowflake(
    user="adelino.dossantos@hotelbeds.com",
    role="datascience_admin",
    warehouse="shared",
)

# ------------------- Preprocessing -------------------

# df = connection.collect_data_snowflake(
#     """
#     WITH base AS (
#         SELECT customer_reporting_level
#              , id_customer_branch
#              , date_trunc('month', date)                                  date_month
#              , month(date)                                                month_number
#              , max(customer_rl_name)                                      customer_rl_name
#              , max(customer_rl_country_name)                              customer_rl_country_name
#              , max(customer_rl_bl_retail)                                 customer_rl_bl_retail
#              , max(customer_rl_wholesale)                                 customer_rl_wholesale
#              , max(customer_reporting_level_commercial_area)              customer_reporting_level_commercial_area
#              , max(customer_connectivity_web)                             customer_connectivity_web
#              , max(customer_connectivity_xml)                             customer_connectivity_xml
#              , max(customer_brand_hotelbeds)                              customer_brand_hotelbeds
#              , max(customer_brand_bedsonline)                             customer_brand_bedsonline
#              , max(customer_brand_hotelopia)                              customer_brand_hotelopia
#              , max(customer_brand_hotelextras)                            customer_brand_hotelextras
#              , max(customer_brand_ds)                                     customer_brand_ds
#              , sum(rn)                                                    rn
#              , sum(searches_total)                                        searches_total
#              , sum(valuations_total)                                      valuations_total
#              , sum(bookings_total)                                        bookings_total
#              , sum(searches_failed) / nullifzero(sum(searches_total))     searches_failed_pct
#              , sum(valuations_failed) / nullifzero(sum(valuations_total)) valuations_failed_pct
#              , sum(bookings_failed) / nullifzero(sum(bookings_total))     bookings_failed_pct
#              , sum(bookingcancellation)                                   bookingcancellation
#              , sum(bookingcancellation) / nullifzero(sum(bookings_total)) cancellation_v_booking_ratio
#              // Incorrect handling, fix later
#              , avg(median_search_process_time)                            median_search_process_time
#              // Incorrect handling, fix later
#              , avg(pctile99_search_process_time)                          pctile99_search_process_time
#              , avg(bookings_with_cases_ops) as bookings_with_cases_ops
#              , avg(bookings_with_cases_sales) as bookings_with_cases_sales
#              , avg(bookings_with_cases_finance) as bookings_with_cases_finance
#              , avg(bookout) as bookout
#              , avg(cases_finance_reason_fm) as cases_finance_reason_fm
#              , avg(cases_finance_status_solved) as cases_finance_status_solved
#              , avg(cases_ops_status_solved) as cases_ops_status_solved
#              , avg(cases_ops_status_noproceed) as cases_ops_status_noproceed
#              , avg(cases_sales_status_solved) as cases_sales_status_solved
#              , avg(bookingcancellationrequest) as bookingcancellationrequest
#              , avg(bookingcancellationandorwaiverrequest) as bookingcancellationandorwaiverrequest
#              , avg(bookingnotfoundorbookingdiscrepancies) as bookingnotfoundorbookingdiscrepancies
#              , avg(bookingpaidatdestination) as bookingpaidatdestination
#              , avg(bookingstatusandbookingreconfirmation) as bookingstatusandbookingreconfirmation
#              , avg(breachofcontract) as breachofcontract
#              , avg(businessmodelcommission) as businessmodelcommission
#              , avg(cancellation) as cancellation
#              , avg(clientnoshow) as clientnoshow
#              , avg(commercialactionsgestures) as commercialactionsgestures
#              , avg(commercialrequestsupport) as commercialrequestsupport
#              , avg(compensationissues) as compensationissues
#              , avg(connectivity) as connectivity
#              , avg(contenterror) as contenterror
#              , avg(contentweberrororsystemfailure) as contentweberrororsystemfailure
#              , avg(contracterror) as contracterror
#              , avg(contractunification) as contractunification
#              , avg(creation) as creation
#              , avg(creditcardfraudccf) as creditcardfraudccf
#              , avg(creditcontrol) as creditcontrol
#              , avg(dissatisfaction) as dissatisfaction
#              , avg(doesnotcorrespondtoagency) as doesnotcorrespondtoagency
#              , avg(doublechargeserroneouscharges) as doublechargeserroneouscharges
#              , avg(duplicatedinvoice) as duplicatedinvoice
#              , avg(feesnegotiationrequired) as feesnegotiationrequired
#              , avg(feesnegotiationrequested) as feesnegotiationrequested
#              , avg(fraudbookings) as fraudbookings
#              , avg(fraudulentagencyfag) as fraudulentagencyfag
#              , avg(fraudulentbooking) as fraudulentbooking
#              , avg(hotelincidentduringstay) as hotelincidentduringstay
#              , avg(hotelchangeterminaltoterminaltransfer) as hotelchangeterminaltoterminaltransfer
#              , avg(hotelclosed) as hotelclosed
#              , avg(hotelbedscompanybrandinfo) as hotelbedscompanybrandinfo
#              , avg(hotelbedshotelroomtypedifferences) as hotelbedshotelroomtypedifferences
#              , avg(inflightservice) as inflightservice
#              , avg(inconveniencesduringrentacarpickupreturn) as inconveniencesduringrentacarpickupreturn
#              , avg(inconveniencesduringticketpickup) as inconveniencesduringticketpickup
#              , avg(inconveniencesduringticketutilization) as inconveniencesduringticketutilization
#              , avg(massiveoverselling) as massiveoverselling
#              , avg(noshow) as noshow
#              , avg(others_) as others_
#              , avg(overbooking) as overbooking
#              , avg(paymentissue) as paymentissue
#              , avg(prepayment) as prepayment
#              , avg(pricematch) as pricematch
#              , avg(ratediscrepancy) as ratediscrepancy
#              , avg(ratedisputeorreservationcontracterror) as ratedisputeorreservationcontracterror
#              , avg(ratesandfees) as ratesandfees
#              , avg(requestvalidcreditcard) as requestvalidcreditcard
#              , avg(resendvoucher) as resendvoucher
#              , avg(reservationerrorornotfound) as reservationerrorornotfound
#              , avg(returnoperationproblem) as returnoperationproblem
#              , avg(salessupportwebcredentials) as salessupportwebcredentials
#              , avg(servicedescriptioninnacurate) as servicedescriptioninnacurate
#              , avg(servicenotprovided) as servicenotprovided
#              , avg(stopsales) as stopsales
#              , avg(systemissue) as systemissue
#              , avg(weberrorsystemfailure) as weberrorsystemfailure
#              , avg(groups) as groups
#              , avg(extras) as extras
#              , avg(hotel_contract) as hotel_contract
#              , avg(accomocation) as accomocation
#              , avg(compensation) as compensation
#              , avg(transfer) as transfer
#              , avg(circuits) as circuits
#              , avg(totality_service) as totality_service
#              , avg(excursions) as excursions
#              , avg(specialist_tours) as specialist_tours
#              , avg(other) as other
#              // overrides
#              , max(HAS_OVERRIDE) HAS_OVERRIDE
#              , avg(AMOUNT_OVERRIDE_EUR) AMOUNT_OVERRIDE_EUR
#              , max(OVERRIDE_TRIGGERED) OVERRIDE_TRIGGERED
#              // markup
#              , avg(markup) markup
#              , avg(channel_markup) channel_markup
#              -- , ifnull(markup, max(percentile_50_markup) over
#              --        (partition by customer_calendar.customer_reporting_level,customer_calendar.ID_CUSTOMER_BRANCH)
#              --    ) markup_interp
#              -- , ifnull(channel_markup, max(percentile_50_channel_markup) over
#              --        (partition by customer_calendar.customer_reporting_level,customer_calendar.ID_CUSTOMER_BRANCH)
#              --    ) channel_markup_interp
#              , max(IFNULL(credit_or_prepayment, '')) IN ('C', 'C & P')    payment_credit
#              , max(IFNULL(credit_or_prepayment, '')) IN ('P', 'C & P')    payment_prepayment
#         FROM HBG_DATASCIENCE.SANDBOX_ANALYTICS.FCT_CHURN_MODELLING_DATA
#         WHERE
#             1=1
#             --AND customer_reporting_level_commercial_area = 'Med & MEAI Core'
#             AND customer_rl_country_name = 'SPAIN'
#             AND DATE_MONTH != '2022-03-01'
#             AND customer_reporting_level IN (
#                     SELECT customer_reporting_level
#                     FROM HBG_DATASCIENCE.SANDBOX_ANALYTICS.FCT_CHURN_MODELLING_DATA
#                     GROUP BY customer_reporting_level
#                     HAVING sum(RN) >= 100
#             )
#             AND id_customer_branch IN (
#                     SELECT id_customer_branch
#                     FROM HBG_DATASCIENCE.SANDBOX_ANALYTICS.FCT_CHURN_MODELLING_DATA
#                     GROUP BY id_customer_branch
#                     HAVING sum(RN) >= 40
#             )
#         GROUP BY customer_reporting_level, id_customer_branch, date_month, month_number
#     ),
#
#     final AS (
#         SELECT *
#             , avg(rn) OVER (PARTITION BY customer_reporting_level
#                             ORDER BY date_month
#                             ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
#             ) AVG_MONTHLY_RN_12M
#             , RN = 0 is_churned
#
#         FROM base
#
#     )
#
#     SELECT *
#     FROM final
#     """
# )

# df["CUSTOMER_RL_COUNTRY_NAME"] = df["CUSTOMER_RL_COUNTRY_NAME"].astype("category")
# df["CUSTOMER_REPORTING_LEVEL_COMMERCIAL_AREA"] = df["CUSTOMER_REPORTING_LEVEL_COMMERCIAL_AREA"].astype("category")
#
# # To backup file above
# df.to_pickle("output/df_raw.pickle")

# Or, load existing data
df = pd.read_pickle("output/df_raw.pickle")

# Sort values
df.sort_values(["CUSTOMER_REPORTING_LEVEL", "ID_CUSTOMER_BRANCH", "DATE_MONTH"], inplace=True)

# Create new variables
lag_period = [1, 2, 3]
columns = [
    "RN",
    "SEARCHES_TOTAL",
    "VALUATIONS_TOTAL",
    "BOOKINGS_TOTAL",
    "SEARCHES_FAILED_PCT",
    "VALUATIONS_FAILED_PCT",
    "BOOKINGS_FAILED_PCT",
    "BOOKINGCANCELLATION",
    "CANCELLATION_V_BOOKING_RATIO",
    "MEDIAN_SEARCH_PROCESS_TIME",
    "PCTILE99_SEARCH_PROCESS_TIME",
    "BOOKINGS_WITH_CASES_OPS",
    "BOOKINGS_WITH_CASES_SALES",
    "BOOKINGS_WITH_CASES_FINANCE",
    "BOOKOUT",
    "CASES_FINANCE_REASON_FM",
    "CASES_FINANCE_STATUS_SOLVED",
    "CASES_OPS_STATUS_SOLVED",
    "CASES_OPS_STATUS_NOPROCEED",
    "CASES_SALES_STATUS_SOLVED",
    "BOOKINGCANCELLATIONREQUEST",
    "BOOKINGCANCELLATIONANDORWAIVERREQUEST",
    "BOOKINGNOTFOUNDORBOOKINGDISCREPANCIES",
    "BOOKINGPAIDATDESTINATION",
    "BOOKINGSTATUSANDBOOKINGRECONFIRMATION",
    "BREACHOFCONTRACT",
    "BUSINESSMODELCOMMISSION",
    "CANCELLATION",
    "CLIENTNOSHOW",
    "COMMERCIALACTIONSGESTURES",
    "COMMERCIALREQUESTSUPPORT",
    "COMPENSATIONISSUES",
    "CONNECTIVITY",
    "CONTENTERROR",
    "CONTENTWEBERRORORSYSTEMFAILURE",
    "CONTRACTERROR",
    "CONTRACTUNIFICATION",
    "CREATION",
    "CREDITCARDFRAUDCCF",
    "CREDITCONTROL",
    "DISSATISFACTION",
    "DOESNOTCORRESPONDTOAGENCY",
    "DOUBLECHARGESERRONEOUSCHARGES",
    "DUPLICATEDINVOICE",
    "FEESNEGOTIATIONREQUIRED",
    "FEESNEGOTIATIONREQUESTED",
    "FRAUDBOOKINGS",
    "FRAUDULENTAGENCYFAG",
    "FRAUDULENTBOOKING",
    "HOTELINCIDENTDURINGSTAY",
    "HOTELCHANGETERMINALTOTERMINALTRANSFER",
    "HOTELCLOSED",
    "HOTELBEDSCOMPANYBRANDINFO",
    "HOTELBEDSHOTELROOMTYPEDIFFERENCES",
    "INFLIGHTSERVICE",
    "INCONVENIENCESDURINGRENTACARPICKUPRETURN",
    "INCONVENIENCESDURINGTICKETPICKUP",
    "INCONVENIENCESDURINGTICKETUTILIZATION",
    "MASSIVEOVERSELLING",
    "NOSHOW",
    "OTHERS_",
    "OVERBOOKING",
    "PAYMENTISSUE",
    "PREPAYMENT",
    "PRICEMATCH",
    "RATEDISCREPANCY",
    "RATEDISPUTEORRESERVATIONCONTRACTERROR",
    "RATESANDFEES",
    "REQUESTVALIDCREDITCARD",
    "RESENDVOUCHER",
    "RESERVATIONERRORORNOTFOUND",
    "RETURNOPERATIONPROBLEM",
    "SALESSUPPORTWEBCREDENTIALS",
    "SERVICEDESCRIPTIONINNACURATE",
    "SERVICENOTPROVIDED",
    "STOPSALES",
    "SYSTEMISSUE",
    "WEBERRORSYSTEMFAILURE",
    "GROUPS",
    "EXTRAS",
    "HOTEL_CONTRACT",
    "ACCOMOCATION",
    "COMPENSATION",
    "TRANSFER",
    "CIRCUITS",
    "TOTALITY_SERVICE",
    "EXCURSIONS",
    "SPECIALIST_TOURS",
    "AMOUNT_OVERRIDE_EUR",
    "MARKUP",
]


for col in columns:
    for lag in lag_period:
        df[col + "_LAG_" + str(lag)] = df.groupby("CUSTOMER_REPORTING_LEVEL")[col].shift(lag)

df["AVG_MONTHLY_RN_12M"] = df["AVG_MONTHLY_RN_12M"].astype("float64")  # Not doing this upsets lightgbm
# df["AVG_MARKUP_LAG_1"] = df["AVG_MARKUP_LAG_1"].astype("float64")  # Not doing this upsets lightgbm

# Save a copy for EDA on Tableau
# df.to_csv("output/df_raw.csv")

# Remove columns which could lead to data leakage
# i.e. Columns that represented values of the same month
# Except RN, which we are predicting
df = df.copy()  # Fixing fragments

removal_exceptions = ["IS_CHURNED"]

df.drop(
    [column for column in columns if column not in removal_exceptions],
    axis=1,
    inplace=True,
)

# Remove pre-2021 data for modelling less accurate?
df = df[df.DATE_MONTH >= dt.date(year=2021, month=1, day=1)]


# ------------------- Modelling -------------------

# Setting all of 2021 as training set and 2022 as test set

split_date = dt.date(year=2022, month=1, day=1)
x_train = df[df.DATE_MONTH < split_date].drop(["IS_CHURNED"], axis=1)
x_test = df[df.DATE_MONTH >= split_date].drop(["IS_CHURNED"], axis=1)
y_train = df[df.DATE_MONTH < split_date]["IS_CHURNED"]
y_test = df[df.DATE_MONTH >= split_date]["IS_CHURNED"]

x_train.drop(["CUSTOMER_REPORTING_LEVEL", "DATE_MONTH", "CUSTOMER_RL_NAME", "ID_CUSTOMER_BRANCH"], axis=1, inplace=True)
x_test.drop(["CUSTOMER_REPORTING_LEVEL", "DATE_MONTH", "CUSTOMER_RL_NAME", "ID_CUSTOMER_BRANCH"], axis=1, inplace=True)

# Use Boruta for feature selection

# Boruta doesn't like any rows with null values
# Dropping null values mean that Boruta's assessment on ...
# variables with potentially meaningful null values is no longer accurate
# However it's still useful for other variables
x_train_boruta = x_train.dropna().copy()
y_train_boruta = y_train.filter(items=y_train.index.intersection(x_train_boruta.index), axis=0).copy()

# Boruta doesn't like categorical variables so dropping those variables as well
x_train_boruta.drop(
    ["CUSTOMER_RL_COUNTRY_NAME", "CUSTOMER_REPORTING_LEVEL_COMMERCIAL_AREA"],
    axis=1,
    inplace=True,
)


# Run variable selection with boruta

model_for_boruta = lightgbm.LGBMRegressor(
    # A simplified model with minimal tuning
    learning_rate=0.08,
    n_estimators=500,
    num_leaves=30,
    min_data_in_leaf=20,
    max_depth=8,
    lambda_l1=0.05,
    lambda_l2=0.05,
    feature_fraction=0.6,
    max_bin=300,
    seed=42,
)

# Define Boruta feature selection method
feat_selector = boruta.BorutaPy(model_for_boruta, n_estimators="auto")

# Find and examine all relevant features
feat_selector.fit(x_train_boruta.values, y_train_boruta.values)
print(x_train_boruta.columns[feat_selector.support_])

# Now, include only variables that boruta approved of
# We also give the variables that boruta could not process correctly a benefit of doubt
# .... or at least that was the plan, but SHAP does not like categorical variable either. Commenting out for now
x_train = x_train[
    list(x_train_boruta.columns[feat_selector.support_])
    # + ['CUSTOMER_RL_COUNTRY_NAME', 'CUSTOMER_REPORTING_LEVEL_COMMERCIAL_AREA']
]
x_test = x_test[
    list(x_train_boruta.columns[feat_selector.support_])
    # + ['CUSTOMER_RL_COUNTRY_NAME', 'CUSTOMER_REPORTING_LEVEL_COMMERCIAL_AREA']
]

# Run our model with grid search for hyperparameter tuning
model_lgbm_params = {
    "n_estimators": [40, 60, 80],
    "max_depth": [3, 4],
    "learning_rate": [0.04, 0.05, 0.07],
    "min_data_in_leaf": [20, 25, 30],
    "max_bin": [300, 400, 550],
    "feature_fraction": [0.35, 0.4, 0.45],
    "num_leaves": [15, 20, 30],
    "lambda_l1": [0, 0.05, 0.075],
    "lambda_l2": [0, 0.05, 0.075],
    "bagging_fraction ": [0.6, 0.75, 1],
    "bagging_freq": [5, 10],
}

# To do: Might need to evaluate objective function later
model_lgbm_grid = sk_ms.RandomizedSearchCV(
    lightgbm.LGBMRegressor(),
    model_lgbm_params,
    cv=3,
    n_iter=40,
    verbose=0,
    return_train_score=True,
)
model_lgbm_grid.fit(x_train, y_train)

model_eval = (
    pd.DataFrame(model_lgbm_grid.cv_results_["params"])
    .merge(
        pd.DataFrame(
            model_lgbm_grid.cv_results_["mean_train_score"],
            columns=["mean_train_score"],
        ),
        left_index=True,
        right_index=True,
    )
    .merge(
        pd.DataFrame(model_lgbm_grid.cv_results_["mean_test_score"], columns=["mean_test_score"]),
        left_index=True,
        right_index=True,
    )
)

print(model_eval)
# model_eval.to_clipboard()

print(model_lgbm_grid.best_params_)
print(model_lgbm_grid.score(x_train, y_train))
print(model_lgbm_grid.score(x_test, y_test))
print(sk_mtrc.confusion_matrix(y_test, model_lgbm_grid.predict(x_test) >= 0.5))

# Refitting the model as Shap does not like RandomizedSearchCV class
model_lgbm = lightgbm.LGBMRegressor(
    **model_lgbm_grid.best_params_,
    seed=42,
)
model_lgbm.fit(x_train, y_train)


# ------------------- Explainability -------------------

shap_explainer = shap.TreeExplainer(model_lgbm)
shap_values = shap_explainer(df[x_train.columns][0:1000])

# shap.summary_plot(shap_values, df[x_train.columns])

shap.plots.bar(shap_values)
shap.plots.bar(shap_values.abs.max(0))

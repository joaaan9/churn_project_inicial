import numpy as np
from scipy.stats import nbinom

import project.util.snowflake as sf

# An example using poisson/negative distribution to test outliers


def convert_params(mu, theta):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    r = theta
    var = mu + 1 / r * mu**2
    p = (var - mu) / var
    return r, 1 - p


def pmf(counts, mu, theta):
    return nbinom.pmf(counts, *convert_params(mu, theta))


def cdf(counts, mu, theta):
    return nbinom.cdf(counts, *convert_params(mu, theta))


connection = sf.Snowflake()

df = connection.collect_data_snowflake(
    """
    WITH
    main AS (
        SELECT customer_reporting_level
            , customer_rl_name
            , customer_rl_country_name
            , customer_rl_bl_retail
            ,customer_rl_wholesale
            , customer_reporting_level_commercial_area
            , customer_connectivity_web
            , customer_connectivity_xml
            , customer_brand_hotelbeds
            , customer_brand_bedsonline
            , customer_brand_hotelopia
            , customer_brand_hotelextras
            , customer_brand_ds
             , date
             , rn
             , sum(IFF(date between '2021-06-01' and '2021-11-30', rn, 0))
                   OVER (partition by customer_reporting_level) rn_last8to2months
             --, sum(IFF(date between '2021-12-01' and '2022-01-31', rn, 0))
             --      OVER (partition by customer_reporting_level) rn_last2months
             , IFF(
                rn > 0,
                0,
                sum(IFF(rn = 0, 0, 1)) over (partition by customer_reporting_level order by date)
            ) as                                                grp
             --, IFF(rn > 0, 1, 0) has_trade,
             --, lag(has_trade) OVER (partition by customer_reporting_level order by date)
        FROM HBG_DATASCIENCE.SANDBOX_ANALYTICS.FCT_CHURN_MODELLING_DATA
             --WHERE date between '2021-06-01' and '2022-01-31'
        WHERE date between '2021-06-01' and '2021-11-30'
              --  AND customer_reporting_level = '0012p00002HGvnlAAD'
              QUALIFY rn_last8to2months != 0
        ORDER BY customer_reporting_level, date
    ),

    main_with_non_trading_days AS (
        SELECT *,
               iff(
                       rn = 0 and sum(rn) over (partition by customer_reporting_level order by date) > 0
                   , sum(1) over (partition by customer_reporting_level, grp order by date)
                   , 0
                   ) as consecutive_zero_trading_days
        FROM main
        ORDER BY customer_reporting_level, date
    )

    -- SELECT
    --     customer_reporting_level
    --     , customer_rl_name
    --     , customer_rl_country_name
    --     , customer_rl_bl_retail
    --     , customer_rl_wholesale
    --     , customer_reporting_level_commercial_area
    --     , customer_connectivity_web
    --     , customer_connectivity_xml
    --     , customer_brand_hotelbeds
    --     , customer_brand_bedsonline
    --     , customer_brand_hotelopia
    --     , customer_brand_hotelextras
    --     , customer_brand_ds
    --     , avg(rn) avg_rn
    --     , sum(rn) sum_rn
    --     , max(consecutive_zero_trading_days) max_consecutive_zero_trading_days
    -- FROM main_with_non_trading_days
    -- GROUP BY
    --     customer_reporting_level
    --     , customer_rl_name
    --     , customer_rl_country_name
    --     , customer_rl_bl_retail
    --     , customer_rl_wholesale
    --     , customer_reporting_level_commercial_area
    --     , customer_connectivity_web
    --     , customer_connectivity_xml
    --     , customer_brand_hotelbeds
    --     , customer_brand_bedsonline
    --     , customer_brand_hotelopia
    --     , customer_brand_hotelextras
    --     , customer_brand_ds
    -- ORDER BY avg_rn desc
    --
    -- ;

    SELECT
        customer_reporting_level
        , customer_rl_name
        , customer_rl_country_name
        , date
        , rn
        , avg(rn) OVER (
            PARTITION BY customer_reporting_level
            ORDER BY date ROWS BETWEEN 84 PRECEDING AND 1 PRECEDING
        ) AS AVG_RN_12W
        , stddev_samp(rn) OVER (
            PARTITION BY customer_reporting_level
            ORDER BY date ROWS BETWEEN 84 PRECEDING AND 1 PRECEDING
        ) AS STD_RN_12W
        , avg(rn) OVER (
            PARTITION BY customer_reporting_level, dayname(date)
            ORDER BY date ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS AVG_RN_WEEKDAY12W
        , stddev_samp(rn) OVER (
            PARTITION BY customer_reporting_level, dayname(date)
            ORDER BY date ROWS BETWEEN 12 PRECEDING AND 1 PRECEDING
        ) AS STD_RN_WEEKDAY12W
        , approx_percentile(rn, 0.8) OVER (
            PARTITION BY customer_rl_country_name, customer_rl_bl_retail, customer_rl_wholesale, date
        ) AS P80_RN_COUNTRY
    --FROM HBG_DATASCIENCE.SANDBOX_ANALYTICS.FCT_CHURN_MODELLING_DATA
    --FROM HBG_DATASCIENCE.SANDBOX_ANALYTICS.FCT_CHURN_MODELLING_DATA_TABLEAUTEST
    FROM main_with_non_trading_days
    WHERE 1=1
        AND rn_last8to2months > 10
        AND customer_rl_country_name IN ('SPAIN')
    QUALIFY 1=1
        AND avg_rn_12w is not null
        AND avg_rn_weekday12w is not null
        AND std_rn_12w is not null
        AND std_rn_weekday12w is not null
    ORDER BY customer_reporting_level, date
    """
)

df = df.astype(
    {
        "CUSTOMER_REPORTING_LEVEL": "str",
        "CUSTOMER_RL_NAME": "str",
        "CUSTOMER_RL_COUNTRY_NAME": "str",
        "DATE": "datetime64[ns]",
        "RN": "int32",
        "AVG_RN_12W": "float32",
        "STD_RN_12W": "float32",
        "AVG_RN_WEEKDAY12W": "float32",
        "STD_RN_WEEKDAY12W": "float32",
        "P80_RN_COUNTRY": "float32",
    }
)

# df['CDF_RN_12W_POISSON'] =
# scipy.stats.nbinom.cdf(df['AVG_RN_12W'],
#                        (df['AVG_RN_12W'] ** 2) / ((df['STD_RN_12W'] ** 2) - (df['AVG_RN_12W'])),
#                        df['AVG_RN_12W']/(df['STD_RN_12W'] ** 2))


df["CDF_RN_12W_NBINOM"] = cdf(df["RN"], df["AVG_RN_12W"], df["STD_RN_12W"])
df["CDF_RN_WEEKDAY12W_NBINOM"] = cdf(df["RN"], df["AVG_RN_WEEKDAY12W"], df["STD_RN_WEEKDAY12W"])

df["CDF_RN_WEEKDAY12W_NBINOM_FLOOR"] = [
    cdf_rn if avg_rn >= 5 else None for avg_rn, cdf_rn in zip(df["AVG_RN_12W"], df["CDF_RN_WEEKDAY12W_NBINOM"])
]

df["AVG_COUNTRY_DAY_PROB"] = df.groupby(["CUSTOMER_RL_COUNTRY_NAME", "DATE"])[
    "CDF_RN_WEEKDAY12W_NBINOM_FLOOR"
].transform(np.median)

df["CDF_RN_12W_NBINOM_FLAG"] = df["CDF_RN_12W_NBINOM"] <= 0.01
df["CDF_RN_WEEKDAY12W_NBINOM_FLAG"] = df["CDF_RN_WEEKDAY12W_NBINOM"] <= 0.01
df["AVG_COUNTRY_DAY_PROB_FLAG"] = df["AVG_COUNTRY_DAY_PROB"] >= 0.01

df.to_csv("./output/stats_methodology_test.csv")

test = df[df["CUSTOMER_REPORTING_LEVEL"] == "0012p00003hMH31AAG"]

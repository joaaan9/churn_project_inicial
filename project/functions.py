from enum import Enum
from typing import Optional


class BaseFunctions(Enum):
    SUM = "sum"
    AVG = "mean"
    MIN = "min"
    MAX = "max"
    ISNULL = "isnull"


class ApplyFunctions(Enum):
    DIFF_DATE_MIN = "diff_date_min"
    DIFF_DATE_MAX = "diff_date_max"


def apply_functions(df, column, function, context: Optional[dict] = None):
    if function == ApplyFunctions.DIFF_DATE_MIN:
        result = df.loc[~df[column].isna(), ["PK", column]].groupby("PK").agg({column: min})
        result = result[column].apply(lambda x: (context["date"][0] - x).days)
        return result
    elif function == ApplyFunctions.DIFF_DATE_MAX:
        result = df.loc[~df[column].isna(), ["PK", column]].groupby("PK").agg({column: max})
        result = result[column].apply(lambda x: (context["date"][0] - x).days)
        return result

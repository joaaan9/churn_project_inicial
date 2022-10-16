import pandas as pd
import snowflake.connector
import sqlalchemy
from dotenv import load_dotenv
import os
from snowflake.connector.pandas_tools import pd_writer


load_dotenv()


class Snowflake:
    def __init__(self):
        self.account = os.environ.get("SNOWFLAKE_ACCOUNT")
        self.region = os.environ.get("SNOWFLAKE_REGION")
        self.user = os.environ.get("SNOWFLAKE_USER")
        self.warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE")
        self.role = os.environ.get("SNOWFLAKE_ROLE")
        self.database = os.environ.get("SNOWFLAKE_DATABASE")
        self.schema = os.environ.get("SNOWFLAKE_SCHEMA")
        self.cnx = snowflake.connector.connect(
            account=self.account,
            region=self.region,
            user=self.user,
            warehouse=self.warehouse,
            role=self.role,
            database=self.database,
            schema=self.schema,
            authenticator="externalbrowser",
        )

    def query(self, query_str: str) -> pd.DataFrame:
        print("Query:", query_str)
        return pd.read_sql(query_str, self.cnx)

    def create_engine(self):
        engine = sqlalchemy.create_engine(
            "snowflake://"
            + "{user}@{account}/{db}/{schema}?warehouse={warehouse}&authenticator={authenticator}&role={role}".format(
                user=self.user,
                account=self.account,
                db=self.database,
                schema=self.schema,
                warehouse=self.warehouse,
                role=self.role,
                authenticator="externalbrowser",
            )
        )

        return engine

    def to_sql(self, df, table_name, if_exists="replace"):
        df.to_sql(
            table_name, con=self.create_engine(), index=False, if_exists=if_exists, chunksize=16000, method=pd_writer
        )

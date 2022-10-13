import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
import os


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

if __name__ == '__main__':
    sn = Snowflake()
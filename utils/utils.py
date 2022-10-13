from pathlib import Path, PureWindowsPath
import os

import pandas as pd

from config import config
from utils.sources.amazon import amazons3
from utils.sources.snowflake import Snowflake


def load_data(path, name=""):
    storage = config.get_var("storage")
    model = config.get_var("name")

    if storage == "local":
        return pd.read_pickle(os.path.join(PureWindowsPath(Path(__file__)).parents[2], f"data/{path}/df_{name}{model}_churn.pickle"))

    elif storage == "s3":
        bucket_s3 = config.get_var("bucket")
        s3 = amazons3()
        s3.download_s3(bucket_s3, f"data/{path}/df_{model}_churn.csv")


def load_prediction_data():
    query_prediction = config.get_var("prediction_dataset_query")
    if not query_prediction:
        return load_data("processed", name="test")
    else:
        return load_from_snowflake(query=query_prediction)


def save_data(df, path, name=""):
    storage = config.get_var("storage")
    model = config.get_var("name")

    if storage == "local":
        df.to_pickle(os.path.join(PureWindowsPath(Path(__file__)).parents[2], f"data/{path}/df_{name}{model}_churn.pickle"))

    elif storage == "s3":
        bucket_s3 = config.get_var("bucket")
        s3 = amazons3()
        s3.upload_s3(df, bucket_s3, f"data/{path}/df_{model}_churn.csv")


def load_from_snowflake(query=""):
    sn = Snowflake()
    if query == "":
        query = config.get_var("sql")
    df = sn.query(query)
    return df




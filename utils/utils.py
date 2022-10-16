from pathlib import Path, PureWindowsPath
import os

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from utils.config import config
from utils.sources.amazon import amazons3
from utils.sources.snowflake import Snowflake


def load_data(path, name=""):
    storage = config.get_var("storage")
    model = config.get_var("name")

    if storage == "local":
        return pd.read_pickle(PureWindowsPath(Path(__file__)).parents[1] / f"data/{path}/df_{name}{model}_churn.pickle")

    elif storage == "s3":
        bucket_s3 = config.get_var("bucket")
        s3 = amazons3()
        s3.download_s3(bucket_s3, f"data/{path}/df_{model}_churn.csv")


def load_prediction_data():
    query_prediction = config.get_var("prediction_dataset_query")
    if not query_prediction:
        return load_data("interim", name="scenarios_")
    else:
        return load_from_snowflake(query=query_prediction)


def save_model(mdl):
    storage = config.get_var("storage")
    model_name = config.get_var("mlflow")["registered_model_name"]
    if not model_name:
        directory = "default"
    else:
        directory = model_name
    path = PureWindowsPath(Path(__file__)).parents[1] / f"models/{directory}"
    mlflow.lightgbm.save_model(mdl, path)
    if storage == "s3":
        bucket_s3 = config.get_var("bucket")
        s3 = amazons3()
        s3.upload_model_s3(path, bucket_s3, f"models/{directory}")


def get_model_mlflow():
    model_name = config.get_var("mlflow")["registered_model_name"]
    model_version = config.get_var("mlflow")["model_version"]
    client = MlflowClient()
    model_uri = client.get_model_version_download_uri(model_name, model_version)
    mdl = mlflow.lightgbm.load_model(model_uri)
    return mdl


def save_data(df, path, name="", type="pickle"):
    storage = config.get_var("storage")
    model = config.get_var("name")
    if type == "csv":
        df.to_csv(PureWindowsPath(Path(__file__)).parents[1] / f"data/{path}/df_{name}{model}_churn.csv")
    else:
        if storage == "local":
            df.to_pickle(PureWindowsPath(Path(__file__)).parents[1] / f"data/{path}/df_{name}{model}_churn.pickle")

        elif storage == "s3":
            bucket_s3 = config.get_var("bucket")
            s3 = amazons3()
            s3.upload_s3(df, bucket_s3, f"data/{path}/df_{model}_churn.csv")


def get_path(path, name: str = ""):
    model = config.get_var("name")
    return PureWindowsPath(Path(__file__)).parents[1] / f"data/{path}/df_{name}{model}_churn.csv"


def load_from_snowflake(query=""):
    sn = Snowflake()
    if query == "":
        query = config.get_var("sql")
    df = sn.query(query)
    return df




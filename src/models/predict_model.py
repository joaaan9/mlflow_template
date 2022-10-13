import mlflow
from mlflow.tracking import MlflowClient
from functions import prediction, get_metrics
from utils.config import config
from utils.decorators import mlflow_decorator
from utils.utils import load_prediction_data


def get_model():
    model_name = config.get_var("mlflow")["registered_model_name"]
    model_version = config.get_var("mlflow")["model_version"]
    client = MlflowClient()
    model_uri = client.get_model_version_download_uri(model_name, model_version)
    mdl = mlflow.lightgbm.load_model(model_uri)
    return mdl


@mlflow_decorator
def workflow():
    """
    Loads data from local or from a query specified in config file, and make a prediction with corresponding metrics
    :return:
    """
    df = load_prediction_data()
    mdl = get_model()
    df_pred = prediction(mdl, df)
    metrics = get_metrics(df, df_pred)
    res = []
    res["metrics"] = metrics
    return res


if __name__ == '__main__':
    workflow()

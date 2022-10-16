from functions import prediction
import pandas as pd
from utils.decorators.mlflow_decorator import mlflow_tracking
from utils.utils import load_prediction_data, get_model_mlflow, save_data, get_path


@mlflow_tracking
def workflow():
    """
    Loads data from local or from a query specified in config file, and make a prediction with corresponding metrics
    :return:
    """
    df = load_prediction_data()
    mdl = get_model_mlflow()
    df_pred = prediction(mdl, df)
    save_data(pd.DataFrame(df_pred), "processed", "predict_", type="csv")
    res = {"artifacts": get_path("processed", "predict_")}
    return res


if __name__ == '__main__':
    workflow()

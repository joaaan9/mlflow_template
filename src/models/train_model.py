from src.models.functions import get_metrics, model
from utils.decorators import mlflow_decorator
from utils.utils import load_data, save_data


@mlflow_decorator
def workflow():
    df_today = load_data("interim", name="today")
    df_scenarios = load_data("interim", name="scenarios")

    df, mdl, important_variables, pred_dev, pred_test, pred_all, feat_test = model(df_scenarios)
    metrics = get_metrics(feat_test, pred_test)

    save_data(pred_dev, "processed", name="dev")
    save_data(pred_test, "processed", name="test")
    save_data(pred_all, "processed", name="all")
    save_data(feat_test, "processed", name="feat_test")

    res = []
    res["model"] = mdl
    res["metrics"] = metrics
    return res


if __name__ == '__main__':
    workflow()

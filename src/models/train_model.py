from src.models.functions import get_metrics, model
from utils.decorators.mlflow_decorator import mlflow_tracking
from utils.utils import load_data, save_data, save_model


@mlflow_tracking
def workflow():
    df_today = load_data("interim", name="today_")
    df_scenarios = load_data("interim", name="scenarios_")

    df, mdl, important_variables, pred_dev, pred_test, pred_all = model(df_scenarios)
    metrics = get_metrics(pred_all["target"], pred_all["pred_target"])

    save_model(mdl)

    save_data(pred_dev, "processed", name="dev_")
    save_data(pred_test, "processed", name="test_")
    save_data(pred_all, "processed", name="all_")

    res = {"model": mdl, "metrics": metrics}
    return res


if __name__ == '__main__':
    workflow()

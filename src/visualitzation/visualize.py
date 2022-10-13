from src.models.predict_model import get_model
from src.visualitzation.utils import print_roc_auc, print_feature_importance
from utils.utils import load_data


def workflow():
    model = get_model(config.get_value("registered_model_name"))
    pred_dev = load_data("interim", name="pred_dev")
    pred_test = load_data("interim", name="pred_test")
    pred_all = load_data("interim", name="pred_all")
    print_roc_auc(
        [
            {"prediction": pred_dev, "curve_name": "TRAIN"},
            {"prediction": pred_test, "curve_name": "TEST"},
            {"prediction": pred_all, "curve_name": "ALL"},
        ]
    )
    print_feature_importance(model)


if __name__ == '__main__':
    workflow()

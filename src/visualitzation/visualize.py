from src.visualitzation.utils import print_feature_importance, print_confusion_matrix, print_roc_auc
from utils.utils import get_model_mlflow, load_data


def workflow():
    model = get_model_mlflow()
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
    print_confusion_matrix(pred_all, threshold=0.6)


if __name__ == '__main__':
    workflow()
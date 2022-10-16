from sklearn.metrics import f1_score, roc_auc_score, recall_score, accuracy_score
from src.models.functions_utils import mdl_pred, mdl_fit, split_data
from utils.config import config


def model(df):
    # Get extra config params
    extra = config.get_var("extra")[0]

    targets_fields = []

    # Split data into train and test
    feat_dev, target_dev, feat_test, target_test = split_data(
        df, "", targets_fields, 0.3
    )


    # Train model
    mdl = mdl_fit(feat_dev, target_dev)

    # Make some predictions on the datasets
    pred_dev = mdl_pred(mdl, feat_dev, target_dev)
    pred_test = mdl_pred(mdl, feat_test, target_test)
    pred_all = mdl_pred(mdl, df, df[""])

    return df, mdl, pred_dev, pred_test, pred_all


def get_metrics(df, df_pred):
    result = {}
    result["f1_score"] = f1_score(df, df_pred)
    result["roc_auc_score"] = roc_auc_score(df, df_pred)
    result["recall_score"] = recall_score(df, df_pred)
    result["accuracy_score"] = accuracy_score(df, df_pred)
    return result


def prediction(model, df):
    important_variables = get_important_variables()
    return model.predict(df[important_variables])

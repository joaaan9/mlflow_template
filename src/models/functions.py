from mlflow import sklearn

from utils import utils


def model(df):
    # Get extra config params

    # Get target fields
    features_fields = []
    targets_fields = []
    for element in df.columns.to_list():
        if element[:3] == "FF_":
            features_fields.append(element)
        else:
            targets_fields.append(element)

    # Split data into train and test
    feat_dev, target_dev, feat_test, target_test = function_utils.split_data(
        df, f"", targets_fields, 0.3
    )

    # Train model
    mdl = utils.mdl_fit(feat_dev, target_dev)

    # Make some predictions on the datasets
    pred_dev = utils.mdl_pred(mdl, feat_dev, target_dev)
    pred_test = utils.mdl_pred(mdl, feat_test, target_test)
    pred_all = utils.mdl_pred(mdl, df, df[f""])

    return df, mdl, pred_dev, pred_test, pred_all, feat_test


def get_metrics(df, df_pred):
    result = []
    result["f1_score"] = sklearn.metrics.f1_score(df, df_pred)
    result["roc_auc_score"] = sklearn.metrics.roc_auc_score(df, df_pred)
    result["recall_score"] = sklearn.metrics.recall_score(df, df_pred)
    result["accuracy_score"] = sklearn.metrics.accuracy_score(df, df_pred)
    return result


def prediction(model, df):
    return model.predict(df)

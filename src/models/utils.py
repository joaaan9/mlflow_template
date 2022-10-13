from lightgbm import LGBMClassifier
import boruta


def find_correlated_variables(feat, cutoff):
    """
    Find highly correlated variables to facilitate variable selection process
    :param feat: dataframe
    :param cutoff: the threshold to cut
    :return: A list of correlated variables.
    It's sorted at the end of function to keep always first the most important features
    """
    cor = feat.corr()
    c1 = cor.stack().drop_duplicates()
    high_cor = c1[c1.values != 1]
    high_cor = high_cor[high_cor > cutoff]

    # Translate result into dataframe
    df_high_cor = pd.DataFrame(high_cor).reset_index()
    df_high_cor.columns = ["variable_1", "variable_2", "correlation"]
    df_high_cor = df_high_cor.sort_values(by="correlation", ascending=False)

    return df_high_cor


def find_important_variables(feat, target):
    """
    Perform feature selection via Boruta using a predefined, simple model.
    :param feat: dataframe of features
    :param target: series of target variables
    :return: A list of variables deemed useful by boruta
    """

    model_for_boruta = LGBMClassifier(
        # A simplified model with minimal tuning
        learning_rate=0.08,
        n_estimators=200,
        num_leaves=30,
        min_data_in_leaf=20,
        max_depth=6,
        lambda_l1=0.5,
        lambda_l2=0.5,
        feature_fraction=0.6,
        max_bin=300,
        seed=42,
    )

    # Define Boruta feature selection method
    feat_selector = boruta.BorutaPy(model_for_boruta, n_estimators="auto")
    feat_selector.fit(feat.values, target)

    return feat.columns[feat_selector.support_]


def mdl_fit(feat, target, params=None, used_features=None):
    if params is None:
        mdl = LGBMClassifier()
    else:
        mdl = LGBMClassifier(**params)

    if used_features is None:
        used_features = feat.columns

    mdl.fit(feat.loc[:, used_features], target)  # @TODO revisar categorical?? , categorical_feature=categoricals)
    return mdl


def mdl_pred(mdl, feat, target, target_col=None):
    if not isinstance(target, pd.DataFrame):
        res = target.to_frame("target")
    else:
        res = target[target_col].to_frame("target")

    res["proba"] = mdl.predict_proba(feat[mdl.feature_name_])[:, 1]
    res["pred_target"] = mdl.predict(feat[mdl.feature_name_])
    return res


def split_data(df, target, excluded_fields, percentage=0.3):
    """
    To split the data with dev and test
    Use two most recent slots as test set, everything else as dev set
    """
    month_list = list(df.SCENARIO_DATE.unique())
    test_months = heapq.nsmallest(2, month_list)
    dev_months = [month for month in month_list if month not in test_months]

    df_dev = df[df.SCENARIO_DATE.isin(dev_months)]
    df_test = df[df.SCENARIO_DATE.isin(test_months)]

    target_dev = df_dev[target]
    target_test = df_test[target]

    feat_dev = df_dev.loc[:, [i for i in df_dev.columns if i not in excluded_fields + [target]]].copy()
    feat_test = df_test.loc[:, [i for i in df_test.columns if i not in excluded_fields + [target]]].copy()

    return feat_dev, target_dev, feat_test, target_test

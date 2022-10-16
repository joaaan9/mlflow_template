import numpy as np
from matplotlib import pyplot as plt
from numpy import mean, std
from pandas_profiling import ProfileReport
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

from utils.config import config
import pandas as pd

def distribution(df):
    """
    Check the distribution, how many hotels or clients are needed to cover the 50% of the GS/GCC ?
    """

    p = (
        df.loc[
            df["FF_ONE_OM_SUM_L13S"] > 0.005,
            ["PK", "FF_ONE_OM_SUM_L13S"],
        ]
        .sort_values(by=["FF_ONE_OM_SUM_L13S"], ascending=False)
        .reset_index(drop=True)
    )

    p["CUM_ONE_OM_SUM_L13S"] = p["FF_ONE_OM_SUM_L13S"].cumsum()
    p["%_cum"] = p["CUM_ONE_OM_SUM_L13S"] / sum(p["FF_ONE_OM_SUM_L13S"])
    plt.figure()
    n = 50000
    p["%_cum"].head(n).plot(title="TOP: " + str(n) + "  out of total of: " + str(p.shape[0]))
    return plt


def print_cross_validation(mdl, feat, target, cv, title):
    scores = cross_val_score(mdl, feat, target, cv=cv, scoring="roc_auc")
    print(scores)
    m_scores = round(mean(scores), 3)
    std_scores = round(std(scores), 3)
    print(title + " Accuracy: %.3f (%.3f)" % (m_scores, std_scores))
    return m_scores, std_scores


def print_feature_importance(mdl):
    """
    Print the features importances sorted from more important to less important
    """
    title = "Feature importance"
    importances = pd.Series(mdl.feature_importances_, index=mdl.feature_name_)
    importances = importances.sort_values(ascending=False)

    fig, ax = plt.subplots()
    importances.plot.bar(ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return plt


def print_roc_auc(listdict):
    model = config.get_value("name")
    title = f"Churn {model} - ROC Curve"
    plt.figure()
    lw = 2
    plt.plot([0, 1], [0, 1], color=plt.cm.tab10.colors[0], lw=lw, linestyle="--")

    for n, i in enumerate(listdict):
        fpr, tpr, thresholds = roc_curve(i["prediction"]["target"], i["prediction"]["proba"])
        fpr_to_plot, tpr_to_plot, thresholds_to_plot = points_to_plot(
            fpr, tpr, thresholds, threshold_to_find=[0.25, 0.5, 0.7]
        )
        plt.scatter(fpr_to_plot, tpr_to_plot, color=plt.cm.tab10.colors[n + 1])
        for j in range(len(thresholds_to_plot)):
            plt.annotate(
                round(thresholds_to_plot[j], 2),
                (fpr_to_plot[j] + 0.02, tpr_to_plot[j] - 0.02),
                color=plt.cm.tab10.colors[n + 1],
            )

        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            color=plt.cm.tab10.colors[n + 1],
            lw=lw,
            label="ROC %s (AUC = %0.2f)" % (i["curve_name"], roc_auc),
        )

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    return plt, roc_auc


def print_confusion_matrix(prediction, title="Confusion matrix", threshold=0.5):
    prediction["binarized_boolean"] = prediction["proba"] >= threshold
    cm = confusion_matrix(prediction["target"], prediction["binarized_boolean"])
    _ = ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{title} - {threshold=}")
    return plt, cm


def find_nearest(array, value):
    """
    Find the index of the value in the array nearest of the parameter value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def points_to_plot(fpr, tpr, thresholds, threshold_to_find):
    """
    Find the closest values from the thresholds_to_find in thresholds and plot the points with the
    x value of FPR and y value of TPR
    It provides us to see by threshold the point in the AUC curve
    """
    fpr_to_plot = []
    tpr_to_plot = []
    thresholds_to_plot = []
    for element in threshold_to_find:
        index = find_nearest(thresholds, element)
        fpr_to_plot.append(fpr[index])
        tpr_to_plot.append(tpr[index])
        thresholds_to_plot.append(thresholds[index])
    return fpr_to_plot, tpr_to_plot, thresholds_to_plot


def create_profile_report(df, filename, title, minimal=True):
    """
    It's used in the model to se the profiling.
    Run using:
    create_profile_report(df,"hotel_churn_profile","hotel_churn_profile",minimal=True)
    create_profile_report(df,"hotel_churn_profile_extend","hotel_churn_profile_extend",minimal=False)
    """
    print("Creating profile report: %s.html" % filename)
    profile = ProfileReport(df.reset_index(drop=True), title=title, minimal=minimal)
    profile.to_file("%s.html" % filename)





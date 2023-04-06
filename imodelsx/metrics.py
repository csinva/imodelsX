from functools import partial

import numpy as np
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             brier_score_loss, mean_squared_error,
                             precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score)


def gini_score(y_true, y_pred):
    """Purer (more accurate) is better"""
    y_pred = y_pred.astype(bool)
    y_pred_sum = y_pred.sum()
    if y_pred_sum == 0 or y_pred_sum == y_pred.size:
        y_mean = y_true.mean()
    else:
        y_mean = y_true[y_pred].mean()
    return gini_binary(y_mean)


def gini_binary(y_mean: float) -> float:
    """{0, 1} -> 1
    {0.5} -> 0.5
    """
    return y_mean ** 2 + (1 - y_mean) ** 2


def entropy_binary(y_mean: float) -> float:
    return -y_mean * np.log2(y_mean) - (1 - y_mean) * np.log2(1 - y_mean)


def auprc_score(y_true, y_pred):
    """area under precision recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)


metrics_classification_discrete = {
    "accuracy": accuracy_score,
    "precision": partial(precision_score, zero_division=0),
    "recall": partial(recall_score, zero_division=0),
    "balanced_accuracy": balanced_accuracy_score,
}
metrics_classification_proba = {
    "roc_auc": roc_auc_score,
    "brier_score_loss": brier_score_loss,
    "auprc": auprc_score,
}
metrics_regression = {
    "r2": r2_score,
    "mse": mean_squared_error,
    "corr": lambda y_true, y_pred: np.corrcoef(y_true, y_pred)[0, 1],
}

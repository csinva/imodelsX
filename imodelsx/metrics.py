import numpy as np

def gini_score(y_true, y_pred):
    """Purer (more accurate) is better
    """
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
    return y_mean**2 + (1 - y_mean)**2

def entropy_binary(y_mean: float) -> float:
    return -y_mean * np.log2(y_mean) - (1 - y_mean) * np.log2(1 - y_mean)

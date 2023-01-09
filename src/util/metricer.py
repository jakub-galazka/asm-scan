import numpy as np

from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

# The method calculates AUROC, AP and Rc|FPR metrics
def calculate_metrics(y_true: list[int], y_pred: list[float]) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64], np.ndarray[np.float64]]:
    roc_auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    # Rc|FPR
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    # Determination of fpr parameter for interpolation of the tpr parameter (based on the number of samples)
    samples = len(y_true)
    order_of_magnitude = int(np.log10(1 / samples))
    fpr_interp = [np.power(10.0, i) for i in range(-1, order_of_magnitude - 1, -1)]

    tpr = np.interp(fpr_interp, fpr, tpr)

    return roc_auc, pr_auc, tpr

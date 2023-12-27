"""
https://github.com/ys-zong/MEDFAIR/blob/main/utils/evaluation.py
"""

import numpy as np
from fairlearn.metrics import (
    demographic_parity_ratio,
    demographic_parity_difference,
    equalized_odds_ratio,
    equalized_odds_difference,
)


def conditional_errors_binary(preds, labels, attrs):
    """
    Compute the conditional errors of A = 0/1. All the arguments need to be one-dimensional vectors.
    :param preds: The predicted label given by a model.
    :param labels: The groundtruth label.
    :param attrs: The label of sensitive attribute.
    :return: Overall classification error, error | A = 0, error | A = 1.
    """

    assert preds.shape == labels.shape and labels.shape == attrs.shape
    cls_error = 1 - np.mean((preds == labels).astype("float"))
    idx = attrs == 0
    error_0 = 1 - np.mean((preds[idx] == labels[idx]).astype("float"))
    error_1 = 1 - np.mean((preds[~idx] == labels[~idx]).astype("float"))
    return cls_error, error_0, error_1


def cal_eqodd(pred_probs, labels, sensitive_attrs, threshold):
    tol_predicted = (pred_probs > threshold).astype("float")
    sens_idx = sensitive_attrs == 0
    target_idx = labels == 0
    cls_error, error_0, error_1 = conditional_errors_binary(
        tol_predicted, labels, sensitive_attrs
    )
    cond_00 = np.mean((tol_predicted[np.logical_and(sens_idx, target_idx)]))
    cond_10 = np.mean((tol_predicted[np.logical_and(~sens_idx, target_idx)]))
    cond_01 = np.mean((tol_predicted[np.logical_and(sens_idx, ~target_idx)]))
    cond_11 = np.mean((tol_predicted[np.logical_and(~sens_idx, ~target_idx)]))
    return 1 - 0.5 * (np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11))


def equalized_odds(y_pred, y_true, sensitive_attrs, ratio=True):
    if ratio:
        return equalized_odds_ratio(y_true, y_pred, sensitive_features=sensitive_attrs)
    return equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive_attrs)


def demographic_disparity(y_pred, y_true, sensitive_attrs, ratio=True):
    if ratio:
        return demographic_parity_ratio(
            y_true, y_pred, sensitive_features=sensitive_attrs
        )
    return demographic_parity_difference(
        y_true, y_pred, sensitive_features=sensitive_attrs
    )

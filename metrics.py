import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    y_pred = np.array([int(item) for item in y_pred])
    y_true = np.array([int(item) for item in y_true])
    y = 2*y_pred - y_true
    TP = len(y[y == 1])
    TN = len(y[y == 0])
    FP = len(y[y == 2])
    FN = len(y[y == -1])
    recall = TP/(TP + FP)
    precision = TP/(TP + FN)
    f1 = 2/(1/precision + 1/recall)
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    y_pred = np.array([int(item) for item in y_pred])
    y_true = np.array([int(item) for item in y_true])
    y = y_pred - y_true
    s = len(y[y == 0])
    accuracy = s/len(y)
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    a = np.sum((y_pred - y_true)**2)
    b = np.sum((np.mean(y_pred) - y_true)**2)
    return 1 - a/b

def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    mse = np.mean((y_pred - y_true)**2)
    return mse

def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    mae = np.mean(np.abs(y_pred - y_true))
    return mae
    
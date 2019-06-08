import numpy as np


def mean_average_error_percentage(y_true, y_pred):
    percentage = np.abs((y_true - y_pred) / y_true) * 100
    return percentage.mean()
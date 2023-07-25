'''
Library for machine and statisical learning loss functions. These functions are helpful 
when determining a model's predictive power or the overall error of a given system.  

Supported Loss Functions:
1) MSE - Mean Squared Error
2) MAE - Mean Absolute Error
3) RMSE - Root Mean Squared Error
4) HMSE - Heteroskedasticity Adjusted Mean Squared Error
5) HMAE - Heteroskedasticity Adjusted Mean Absolute Error
6) HRMSE - Heteroskedasticity Adjusted Root Mean Squared Error

Future Estimation Algorithms:
1) Huber-Loss 
2) Entropy

'''

import numpy as np
import pandas as pd

import numpy as np

def mse(y_true, y_pred):
    """
    Compute the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Mean Squared Error.
    """
    return np.mean(np.square(y_true - y_pred))

def mae(y_true, y_pred):
    """
    Compute the Mean Absolute Error (MAE) between true and predicted values.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Mean Absolute Error.
    """
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """
    Compute the Root Mean Squared Error (RMSE) between true and predicted values.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Root Mean Squared Error.
    """
    return np.sqrt(mse(y_true, y_pred))

def hmse(y_true, y_pred):
    """
    Compute the Heteroskedasticity Adjusted Mean Squared Error (HMSE) between true and predicted values.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Heteroskedasticity Adjusted Mean Squared Error.
    """
    return np.mean(np.square(1 - (y_true / y_pred)))

def hmae(y_true, y_pred):
    """
    Compute the Heteroskedasticity Adjusted Mean Absolute Error (HMAE) between true and predicted values.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Heteroskedasticity Adjusted Mean Absolute Error.
    """
    return np.mean(np.abs(1 - (y_true / y_pred)))

def hrmse(y_true, y_pred):
    """
    Compute the Heteroskedasticity Adjusted Root Mean Squared Error (HRMSE) between true and predicted values.

    Parameters:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: Heteroskedasticity Adjusted Root Mean Squared Error.
    """
    return np.sqrt(hmse(y_true, y_pred))


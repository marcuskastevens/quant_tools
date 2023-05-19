'''
Library for covariance estimation functions. Contains general wrapper risk 
matrix function, allowing you to run any risk model from one function.

Supported Estimation Algorithms:
1) Empirical/Historical Covariance
2) Shrunken Empirical Covariance
3) Trailing Covariance
4) Shrunken Trailing Covariance
5) Exponentially Weighted Covariance
6) Shrunken Exponentially Weighted Covariance


Future Estimation Algorithms:
1) Ledoit-Wolf Shrunken Covariance
2) De-Toning / Constant Residual Eigenvalue Method (MLDP)
3) RMT / RIE / Marcenko-Pastur EPO Risk Model
4) DDC Multivariate GARCH Covariance

'''

import pandas as pd
import numpy as np

# -------------------------------- Risk Models --------------------------------

def risk_matrix(returns: pd.DataFrame, method="empirical_cov", **kwargs):
    """ Compute the covariance matrix, using the "risk model" supplied in the method parameter.

    Args:
        returns (pd.DataFrame): historical returns.
        method (str, optional): covariance risk model. Defaults to "empirical_cov".
    """

    return risk_models[method](returns=returns, **kwargs)

def empirical_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """ Computes full-sample empirical covariance matrix.

    Args:
        returns (pd.DataFrame): historical returns.

    Returns:
        pd.DataFrame: empirical covariance matrix.
    """

    # Compute empirical covariance matrix
    cov = returns.cov()

    if cov.isnull().values.any():
        raise ValueError("Covariance matrix contains missing values")
    
    return cov

def shrunken_empirical_cov(returns: pd.DataFrame, shrinkage: float = 0.1) -> pd.DataFrame:
    """ Computes shrunken empirical covariance matrix.

    Args:
        returns (pd.DataFrame): historical returns.
        shrinkage (float, optional): percent shrinkage of off-diagonal correlations.

    Returns:
        pd.DataFrame: shrunken empirical covariance matrix.
    """

    # Calculate volatility estimates
    vols = returns.std()

    # Calculate correlation estimates
    corr = returns.corr()

    # Shrink off-diagonal correlations
    shrunken_corr = corr * (1 - shrinkage)
    np.fill_diagonal(shrunken_corr.values, 1)

    # Compute covariance matrix
    cov = corr_to_cov(corr = shrunken_corr, vols = vols)

    if cov.isnull().values.any():
        raise ValueError("Covariance matrix contains missing values")
    
    return cov

def empirical_trailing_cov(returns: pd.DataFrame, lookback_cov: int = 150) -> pd.DataFrame:
    """ Computes "lookback_cov" day empirical covariance matrix.

    Args:
        returns (pd.DataFrame): historical returns.
        lookback_cov (int, optional): covariance matrix with "lookback_cov" day center-of-mass.

    Returns:
        pd.DataFrame: trailing "lookback_cov" day empirical covariance matrix.
    """

    # Compute trailing covariance matrix
    cov = returns.tail(lookback_cov).cov()

    if cov.isnull().values.any():
        raise ValueError("Covariance matrix contains missing values")

    return cov

def shrunken_empirical_trailing_cov(returns: pd.DataFrame, shrinkage: float = 0.1, lookback_cov: int = 150) -> pd.DataFrame:
    """Computes shrunken "lookback_cov" day empirical covariance matrix.

    Args:
        returns (pd.DataFrame): historical returns.
        shrinkage (float, optional): percent shrinkage of off-diagonal correlations.
        lookback_cov (int, optional): covariance matrix with "lookback_cov" day center-of-mass.

    Returns:
        pd.DataFrame: shrunken "lookback_cov" day empirical covariance matrix.
    """

    # Compute trailing returns
    returns = returns.tail(lookback_cov)

    # Calculate volatility estimates
    vols = returns.std()

    # Calculate correlation estimates
    corr = returns.corr()

    # Shrink off-diagonal correlations
    shrunken_corr = corr * (1 - shrinkage)
    np.fill_diagonal(shrunken_corr.values, 1)

    # Compute covariance matrix
    cov = corr_to_cov(corr = shrunken_corr, vols = vols)

    if cov.isnull().values.any():
        raise ValueError("Covariance matrix contains missing values")
    
    return cov


def ewma_cov(returns: pd.DataFrame, lookback_vol: int = 60, lookback_corr: int = 150) -> pd.DataFrame:
    """
    Compute the covariance matrix of returns based on the given methodology.
    
    Args:
        returns (pd.DataFrame): historical returns.
        lookback_vol (int, optional): exponentially-weighted daily returns with a "lookback_vol" day center-of-mass.
        lookback_corr (int, optional): exponentially-weighted 3-day overlapping returns with a "lookback_corr" day center-of-mass.
    
    Returns:
        pd.DataFrame: ewma covariance matrix.
    """

    n = returns.shape[1]

    # Calculate volatility estimates
    vols = returns.ewm(span=lookback_vol, min_periods=lookback_vol).std().iloc[-1]
    
    # Calculate correlation estimates
    returns_3d = returns.rolling(window=3).sum()
    corr = returns_3d.ewm(span=lookback_corr, min_periods=lookback_corr).corr().droplevel(0).iloc[-n:]
    
    # Compute covariance matrix
    cov = corr_to_cov(corr = corr, vols = vols)

    if cov.isnull().values.any():
        raise ValueError("Covariance matrix contains missing values")
    
    return cov

def shrunken_ewma_cov(returns: pd.DataFrame, shrinkage: float = 0.1, lookback_vol: int = 60, lookback_corr: int = 150) -> pd.DataFrame:
    """
    Compute the covariance matrix of returns based on the given methodology.
    
    Args:
        returns (pd.DataFrame): historical returns.
        shrinkage (float, optional): percent shrinkage of off-diagonal correlations.
        lookback_vol (int, optional): exponentially-weighted daily returns with a "lookback_vol" day center-of-mass.
        lookback_corr (int, optional): exponentially-weighted 3-day overlapping returns with a "lookback_corr" day center-of-mass.
    
    Returns:
        pd.DataFrame: ewma covariance matrix of the returns.
    """

    n = returns.shape[1]

    # Calculate volatility estimates
    vols = returns.ewm(span=lookback_vol, min_periods=lookback_vol).std().iloc[-1]
    
    # Calculate correlation estimates
    returns_3d = returns.rolling(window=3).sum()
    corr = returns_3d.ewm(span=lookback_corr, min_periods=lookback_corr).corr().droplevel(0).iloc[-n:]

    # Shrink off-diagonal correlations
    shrunken_corr = corr * (1 - shrinkage)
    np.fill_diagonal(shrunken_corr.values, 1)
    
    # Compute covariance matrix
    cov = corr_to_cov(corr = shrunken_corr, vols = vols)
    
    if cov.isnull().values.any():
        raise ValueError("Covariance matrix contains missing values")
    
    return cov

# -------------------------------- Utilities --------------------------------

risk_models = { "empirical_cov" : empirical_cov,
                "shrunken_empirical_cov" : shrunken_empirical_cov,
                "empirical_trailing_cov" : empirical_trailing_cov, 
                "shrunken_empirical_trailing_cov" : shrunken_empirical_trailing_cov,
                "ewma_cov" : ewma_cov,
                "shrunken_ewma_cov" : shrunken_ewma_cov
              }

def corr_to_cov(corr: pd.DataFrame, vols: pd.Series) -> pd.DataFrame:
    """ Convert a correlation matrix to a covariance matrix using the given volatility (std) vector.

    Args:
        corr (pd.DataFrame): correlation matrix.
        vols (pd.Series): volatility vector.

    Returns:
        pd.DataFrame: covariance matrix converted from correlation matrix and standard deviations.
    """

    # Element-wise product of vols (e.g., vol1*vol1, vol1*vol2, vol2*vol1, vol2*vol2)
    # Intuitively, compute all possible pairwise products between the elements of two vectors
    vol_product = np.outer(vols, vols)

    # Multiply vol1 * vol2 * corr1,2 for the off-diagonals
    cov = vol_product * corr

    return cov

def cov_to_corr(cov: pd.DataFrame, vols: pd.Series) -> pd.DataFrame:
    """ Convert a covariance matrix to a correlation matrix using the given volatility (std) vector.

    Args:
        cov (pd.DataFrame): covariance matrix.
        vols (pd.Series): volatility vector.

    Returns:
        pd.DataFrame: correlation matrix converted from correlation matrix and standard deviations.
    """

    # Element-wise product of vols (e.g., vol1*vol1, vol1*vol2, vol2*vol1, vol2*vol2)
    # Intuitively, compute all possible pairwise products between the elements of two vectors
    vol_product = np.outer(vols, vols)

    # Divide cov1,2 / vol1 * vol2
    corr = cov / vol_product 

    return corr
'''
Library for portfolio optimization objective functions.

'''

from quant_tools import performance_analysis as pa
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------- Objective Functions -------------------------------------------------------------------------
def sharpe_ratio_obj(w: pd.Series, expected_returns: pd.Series, cov: pd.DataFrame, neg: bool = True) -> float:
    """ Compute portfolio Sharpe Ratio based on portfolio weights, expected returns, 
        and covariance matrix.

    Args:
        w (pd.Series): position weights vector.
        expected_returns (pd.Series): expected returns vector.
        cov (pd.DataFrame): covariance matrix of returns.
        neg (bool, optional): if true, returns neg SR for minimization optimization objectives. Defaults to True.

    Returns:
        float: portfolio SR.
    """
    # Portfolio Expected Return
    mu = (expected_returns.T * w).sum()

    # Portfolio Vol
    sigma = np.sqrt(np.dot(np.dot(w.T, cov), w))

    # Portfolio SR
    sharpe_ratio = mu / sigma * 252 ** .5
    
    if neg == True: 
        return -sharpe_ratio
    
    return sharpe_ratio

def risk_averse_return_obj(w: pd.Series, expected_returns: pd.Series, cov: pd.DataFrame, gamma: float = .5) -> float:
    """ Cost function for classic risk averse portfolio based on portfolio weights, expected returns, covariance matrix, 
        and risk aversion.

    Args:
        w (pd.Series): position weights vector.
        expected_returns (pd.Series): expected returns vector.
        cov (pd.DataFrame): covariance matrix of returns.
        gamma (float, optional): risk aversion/cost coefficient.

    Returns:
        float: portfolio SR.
    """
    # Portfolio Expected Return
    mu = (expected_returns.T * w).sum() * 252

    # Portfolio Vol
    sigma = np.sqrt(np.dot(np.dot(w.T, cov), w)) * 252 ** .5

    # Risk Averse Cost Function
    cost = mu - gamma * sigma
    
    return(-cost)

def risk_turnover_averse_return_obj(w0: pd.Series, w1: pd.Series, expected_returns: pd.Series, cov: pd.DataFrame, gamma: float = .5, delta: float = .5) -> float:
    """ Cost function for risk and turnover averse portfolio based on portfolio weights, expected returns, covariance matrix, 
        risk aversion, and turnover cost.

    Args:
        w (pd.Series): position weights vector.
        expected_returns (pd.Series): expected returns vector.
        cov (pd.DataFrame): covariance matrix of returns.
        gamma (float, optional): risk aversion/cost coefficient.
        delta (float, optional): turnover aversion/cost coefficient.

    Returns:
        float: portfolio SR.
    """
    # Portfolio Expected Return
    mu = (expected_returns.T * w1).sum() * 252

    # Portfolio Vol
    sigma = np.sqrt(np.dot(np.dot(w1.T, cov), w1)) * 252 ** .5

    # Turnover
    turnover = np.abs(w1 - w0)

    # Risk & Turnover Averse Cost Function
    cost = mu - gamma * sigma - delta * turnover
    
    return(-cost)

def dasr_obj(w: pd.Series, hist_returns: pd.DataFrame, neg: bool = True) ->  float:
    """ Compute portfolio Drift-Adjusted Sharpe Ratio (DASR) based on portfolio weights & 
        expected returns.

    Args:
        w (pd.Series): position weights vector. 
        hist_returns (pd.DataFrame): historical returns matrix of portfolio components.
        neg (bool, optional): if true, returns neg DASR for minimization optimization objectives. Defaults to True.

    Returns:
        float: portfolio DASR.
    """

    # Get portfolio returns
    returns = (hist_returns * w).sum(1)

    # Get weighted portfolio DASR
    portfolio_dasr = pa.drift_adjusted_sharpe_ratio(returns.dropna())

    if neg:
        # Return negative DASR for portfolio optimization
        return (-portfolio_dasr)
    
    # Return real DASR for performance analysis purposes
    return portfolio_dasr

def risk_parity_obj(w: pd.Series, cov: pd.DataFrame) -> float:
    """ Traditional risk parity objective function.
        Minimization ensures portfolio acheives equal variance contribution.       
 
    Args:
        w (pd.Series): portfolio weights vector.
        cov (pd.DataFrame): covariance matrix of returns.

    Returns:
        float: difference between current and equal variance contributions.
    """

    # N portfolio constituents
    n = len(w)

    # Get equal variance contribution weights
    equal_risk_contribution = np.array([1 / n] * n)

    # Get portfolio variance
    variance = w.T.dot(cov).dot(w)

    # Get weighted asbolute risk
    weighted_absolute_risk_contribution = w.T.dot(cov) * w

    # Get each constituent's proportion risk contribution
    risk_contribution = weighted_absolute_risk_contribution / variance

    # Measure the absolute difference between current vs. equal risk contribution
    diff = np.abs((risk_contribution - equal_risk_contribution)).sum()

    return diff 

def dollar_risk_parity_obj(n_units: pd.Series, cov: pd.DataFrame) -> float:
    """ Inspired by CTA and Trend Follwers' risk management practices, the Dollar Risk Parity objective function
        targets "target_risk" percent risk per position (equal risk contribution) which represents 1 SDs of the 
        underlying instrument's price movement. Here, the covariance and variance of each asset defines its risk to account for
        correlation structures and mitiage over exposure to a single risk factor.
               
        Positions represent how many shares to purchase for an 1 SD move in the underlying instrument 
        to represent "target_risk" percent loss in "portfolio_value". This follows the risk management practices of 
        select portfolio managers that target equal risk allocation to each trade/position, but leverages stop-losses 
        (e.g., 1SD stop) to limit exogenous risk exposure.

        This follows the intuition behind traditional equal variance contribution risk parity algorithms, but defines risk as dollar value
        at risk (e.g., stop loss set at 1SD) instead of purely variance.          

    Args:
        n_units (pd.Series): number of units (e.g., shares, contracts) to purchase per portfolio constituent. 
        cov (pd.DataFrame): covariance matrix of portfolio constituents' prices.
        target_risk (float): target percent risk per position.
        portfoio_value (float): portfolio dollar value.

    Returns:
        float
    """

    # N portfolio constituents
    n = len(n_units)

    # Get equal percent risk contribution (representing target 1SD move in each instrument)
    equal_risk_contribution = np.array([1/n] * n) 

    # Get portfolio dollar variance
    variance = n_units.T.dot(cov).dot(n_units)
    
    # Get weighted risk (respective dollar variance)    
    weighted_absolute_risk_contribution = n_units.T.dot(cov) * n_units
    
    # Get percent weighted risk contribution (percent contribution to variance)
    risk_contribution = weighted_absolute_risk_contribution / variance

    # Check if the dollar risk contribution of current portfolio = equal_risk_contribution
    # Measure the absolute difference between current vs. equal risk contribution
    diff = np.abs((risk_contribution - equal_risk_contribution)).sum()

    return diff 

def atr_risk_parity_obj(n_units: pd.Series, cov: pd.DataFrame) -> float:
    """ Inspired by CTA and Trend Follwers' risk management practices, the ATR Risk Parity objective function
        targets equal risk contribution (i.e., (True Range)^2 risk) which represents True Range of the 
        underlying instrument's price movement. Here, the cov represents the parwise relationship between Asset 1's 
        True Range & Asset 2's True Range, defining ATR-derived risk based on the correlation structure of the underlying portfolio
        and mitigating over exposure to a single risk factor.
               
        Positions represent how many shares to purchase for a equal portfolio risk contribution as a function of True Range.
        This follows the risk management practices of select portfolio managers that target equal risk allocation to each trade/position, 
        but leverages stop-losses (e.g., 1 ATR stop) to limit exogenous risk exposure.

        This follows the intuition behind traditional equal variance contribution risk parity algorithms, but defines risk as dollar value
        at risk (e.g., stop loss set at 1 ATR) instead of purely variance.          

    Args:
        n_units (pd.Series): number of units (e.g., shares, contracts) to purchase per portfolio constituent. 
        cov (pd.DataFrame): covariance matrix of portfolio constituents' prices.
        target_risk (float): target percent risk per position.
        portfoio_value (float): portfolio dollar value.

    Returns:
        float
    """

    # N portfolio constituents
    n = len(n_units)

    # Get equal percent risk contribution (representing target 1SD move in each instrument)
    equal_risk_contribution = np.array([1/n] * n) 

    # Get portfolio dollar variance
    portfolio_true_range_risk = n_units.T.dot(cov).dot(n_units)
    
    # Get weighted risk (respective dollar variance)    
    weighted_true_range_risk_contribution = n_units.T.dot(cov) * n_units
    
    # Get percent weighted risk contribution (percent contribution to variance)
    risk_contribution = weighted_true_range_risk_contribution / portfolio_true_range_risk

    # Check if the dollar risk contribution of current portfolio = equal_risk_contribution
    # Measure the absolute difference between current vs. equal risk contribution
    diff = np.abs((risk_contribution - equal_risk_contribution)).sum()

    return diff  
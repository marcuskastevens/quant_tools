import numpy as np
import pandas as pd
from pandas_ta.overlap import ma
from quant_tools import performance_analysis as pa


def scale_vol(strategy_returns: pd.Series, target_vol = .10) -> pd.Series:
    """ Scale strategy returns to a target volatility.

    Args:
        strategy_returns (pd.Series): time-series of returns.
        target_vol (float, optional): targeted volatility for strategy. Defaults to .10.

    Returns:
        pd.Series: volatility scaled strategy returns
    """
    # Use vol_scalar to multiply strategy_returns by to realize target_vol
    vol_scalar = target_vol / pa.vol(strategy_returns = strategy_returns)
    
    # Scale returns
    strategy_returns = strategy_returns * vol_scalar

    return strategy_returns


def volatility_filter(returns: pd.DataFrame, p=0.50, filter_type='high_vol') -> pd.DataFrame:
    """ Drops p percent of given securities based on their volatility ranking (i.e., higher vol gets dropped). 

    Args:
        returns (pd.DataFrame): returns of multiple securities
        filter_type (str): indicates whether to filter high or low volatility.
        daily_ret_threshold (int, optional): proportion of securities to be dropped. Defults to 0.50 or 50%.

    Returns:
        pd.DataFrame: all securities with acceptable volatility. 
    """
    n = int(len(returns.columns)*p)

    # Drop all n largest vol tickers
    vols = returns.std()*252**.5

    if filter_type=='high_vol':
        tickers = vols.nlargest(n).index
    elif filter_type=='low_vol':
        tickers = vols.nsmallest(n).index
    else:
        raise ValueError('filter_type must be either 1) high_vol or 2) low_vol')
            
    print(f'Dropped tickers: {tickers}')

    return returns.drop(columns=tickers)

def true_range(prices):
    """Function to get the var_atr & cov_atr which allows traders to leverage atr in a similar manner to variance & covariance.

    Args:
        prices (_type_): _description_
    """

    high = prices.High
    low = prices.Low
    close = prices["Adj Close"]

    ranges = [high - low, high - close.shift(1), close.shift(1) - low]
    true_range = pd.concat(ranges, axis=1)
    true_range = true_range.abs().max(axis=1)
    
    return true_range

def true_range_covariance(true_ranges: pd.DataFrame, lookback_window=20):
    """Function to get the var_atr & cov_atr which allows traders to leverage atr in a similar manner to variance & covariance.

    Args:
        true_ranges (pd.DataFrame): _description_
        lookback_window (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """   
    sample_true_ranges = true_ranges.tail(lookback_window)
    n = lookback_window
    cov_true_range = sample_true_ranges.T.dot(sample_true_ranges) / (n-1)
    
    return cov_true_range
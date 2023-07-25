import numpy as np
import pandas as pd
from pandas_ta.overlap import ma
from quant_tools import performance_analysis as pa


def ewma(time_series, span: float = None, com: float = None, halflife: float = None, alpha: float = None) -> pd.Series or pd.DataFrame:
    """ Applies an exponentially-weighted moving average according to user-specified decay method.
        
        The parameters "com", "span", "halflife", and "alpha" in the pandas.Series.ewm() function are used 
        to specify the decay or smoothing factor for exponentially weighted calculations. 
        Each parameter represents a different way to express the decay or smoothing factor and are mutually exclusive.

    Args:
        time_series (pd.Series or pd.DataFrame): Time series or matrix of returns, variance, vol, etc.

        span (float): It specifies the decay in terms of the span. The value of span should be a float. 
                The span represents the number of observations to include in the exponential weighting, 
                and it is related to com as span = 1 / (1 - com). A smaller span value gives more weight 
                to recent observations, while a larger span value gives more weight to past observations.

                Decay Alpha = 2 / (span + 1)

        com (float):  Center of Mass - specifies the decay in terms of the center of mass. 
                      The value of com should be a float between 0 and 1. 
                      A smaller com value gives more weight to recent observations, 
                      while a larger com value gives more weight to past observations.

                      Decay Alpha = 1 / (com + 1)

        halflife (float): It specifies the decay in terms of the half-life. The value of halflife can be a 
                          float, string, or timedelta. The half-life represents the time it takes for an 
                          observation to decay to half its value. It is related to com as halflife = ln(2) / ln(1 + com). 
                          Smaller halflife values give more weight to recent observations, while larger halflife values 
                          give more weight to past observations.

                          Decay Alpha = 1 - exp(-ln(2) / halflife) 

        alpha (float): It directly specifies the smoothing factor alpha as a float. 
                       The value of alpha should be between 0 and 1. A smaller alpha value gives more weight to 
                       recent observations, while a larger alpha value gives more weight to past observations.

                       Decay Alpha = alpha where 0 < alpha < 1
    
    Returns: 
        pd.Series or pd.DataFrame: exponentially-weighted moving average of given time series or matrix.
    """

    # Calls pandas ewm method to create a decayed time series and catch all errors.  
    ewma = time_series.ewm(span=span, com=com, halflife=halflife, alpha=alpha).mean()

    return ewma


def scale_vol(returns: pd.Series, target_vol = .10) -> pd.Series:
    """ Scale strategy returns to a target volatility.

    Args:
        returns (pd.Series): time-series of returns.
        target_vol (float, optional): targeted volatility for strategy. Defaults to .10.

    Returns:
        pd.Series: volatility scaled strategy returns
    """
    # Use vol_scalar to multiply returns by to realize target_vol
    vol_scalar = target_vol / pa.vol(returns = returns)
    
    # Scale returns
    returns = returns * vol_scalar

    return returns


def vol_filter(returns: pd.DataFrame, p=0.50, filter_type='high_vol') -> pd.DataFrame:
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
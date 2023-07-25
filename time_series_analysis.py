"""
Library of time series and economitric tools.


"""

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

        span (float): specifies the decay in terms of the span. The span represents the number of
                      observations to include in the exponential weighting, and it is related to com as
                      span = 1 / (1 - com). A smaller span value gives more weight to recent observations,
                      while a larger span value gives more weight to past observations.

                      Decay Alpha = 2 / (span + 1)

        com (float):  center of Mass - specifies the decay in terms of the center of mass. 
                      The value of com should be a float between 0 and 1. 
                      A smaller com value gives more weight to recent observations, 
                      while a larger com value gives more weight to past observations.

                      Decay Alpha = 1 / (com + 1)

        halflife (float): specifies the decay in terms of the half-life. The value of halflife can be a 
                          float, string, or timedelta. The half-life represents the time it takes for an 
                          observation to decay to half its value. It is related to com as halflife = ln(2) / ln(1 + com). 
                          Smaller halflife values give more weight to recent observations, while larger halflife values 
                          give more weight to past observations.

                          Decay Alpha = 1 - exp(-ln(2) / halflife) 

        alpha (float): directly specifies the smoothing factor alpha as a float. 
                       The value of alpha should be between 0 and 1. A smaller alpha value gives more weight to 
                       recent observations, while a larger alpha value gives more weight to past observations.

                       Decay Alpha = alpha where 0 < alpha < 1
    
    Returns: 
        pd.Series or pd.DataFrame: exponentially-weighted moving average of given time series or matrix.
    """

    # Calls pandas ewm method to create a decayed time series and catch all errors.  
    ewma = time_series.ewm(span=span, com=com, halflife=halflife, alpha=alpha).mean()

    return ewma


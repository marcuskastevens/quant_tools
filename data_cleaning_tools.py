import numpy as np
import pandas as pd

def drop_outlier_returns(returns: pd.DataFrame, daily_ret_threshold=20) -> pd.DataFrame:
    """ Drops securities that have abnormal returns. This is a data cleansing excercise that controls for potential data errors or 
        obscure events.

    Args:
        returns (pd.DataFrame): returns of multiple securities
        daily_ret_threshold (int, optional): max daily allowable return across all securities. Defaults to 20.

    Returns:
        pd.DataFrame: all securities with acceptable daily returns. 
    """
    
    # Drop all tickers whose returns were over a particlular threshold in a single day
    drop_index = np.where(returns.abs()>daily_ret_threshold)[1] # get second part of np.array since that indicates the tickers
    
    # Identify tickers to be dropped
    tickers = returns.iloc[:, drop_index].columns
    
    print(f'Dropped tickers: {tickers}')

    return returns.drop(columns=tickers)


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
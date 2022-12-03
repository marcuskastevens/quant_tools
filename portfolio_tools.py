from scipy.optimize import minimize as opt
from scipy.optimize import Bounds
from scipy import stats
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------- Portfolio Optimization -------------------------------------------------------------------------

# Compute portfolio Sharpe Ratio based on asset weights, returns, and covariance matrix
def portfolio_sharpe_ratio(asset_weights, expected_returns, cov_matrix, neg = True):
    """ Compute portfolio Sharpe Ratio based on asset weights, returns, and covariance matrix.

    Args:
        asset_weights (_type_): _description_
        expected_returns (_type_): _description_
        cov_matrix (_type_): _description_
        neg (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    mu = (expected_returns.T * asset_weights).sum()
    sigma = np.sqrt(np.dot(np.dot(asset_weights.T, cov_matrix), asset_weights))
    sharpe_ratio = mu / sigma * 252 ** .5
    
    if neg == True: 
        return -sharpe_ratio
    
    return sharpe_ratio

def mvo(hist_returns: pd.DataFrame, expected_returns: pd.DataFrame, target_vol = .01):
    """ Constrained or unconstrained Mean Variance Optimization. This leverages convex optimization to identify local minima which serve to minimize an objective function.
        In the context of portfolio optimization, our objective function is the negative portfolio SR. 

    Args:
        hist_returns (pd.DataFrame): expanding historical returns of specified asset universe
        expected_returns (pd.DataFrame): expected returns across specified asset universe, normally computed via statistical model
        target_vol (float, optional): targeted ex-ante volatilty based on covariance matrix. Defaults to .10.

    Returns:
        _type_: _description_
    """

    # Match tickers across expected returns and historical returns
    expected_returns.dropna(inplace=True)
    hist_returns = hist_returns.loc[:, expected_returns.index]

    vols = hist_returns.std()*252**.5
    cov_matrix = hist_returns.cov()
    
    n = hist_returns.columns.size
    if n > 0:
        
        # Initial guess is naive 1/n portfolio
        initial_guess = np.array([1 / n] * n)
        
        # Set max allocation per security
        bounds = Bounds(-.2, .2)

        # Ensure full portfolio exposure (100%) and target volatility
        constraints = [{"type": "eq", "fun": lambda vols: np.sum(vols) - 1}, 
                        {"type": "eq", "fun": lambda vols: np.sqrt(np.dot(np.dot(vols.T, cov_matrix), vols)) - target_vol}]

        wts = pd.Series(opt(portfolio_sharpe_ratio, 
                initial_guess,
                args=(expected_returns, cov_matrix), 
                method='SLSQP', 
                bounds = bounds,
                constraints=constraints)['x']
            )

        # Assign weights to assets
        wts.index = vols.index

        # print('Target Vol: ')
        # print(np.sqrt(np.dot(np.dot(wts.T, cov_matrix), wts)))

        # mvo_sr = sharpe_ratio(wts, expected_returns, cov_matrix, neg=False)
        
        return wts
    return

def portfolio_stop_loss(strategy_returns: pd.Series, stop_loss_target = -.01, t_costs = 0, re_entry_eod = True) -> pd.Series:
    """_summary_

    Args:
        strategy_returns (pd.Series): time-series of returns.
        stop_loss_target (float, optional): _description_. Defaults to -.01.
        t_costs (int, optional): _description_. Defaults to 0.
        re_entry_eod (bool, optional): _description_. Defaults to True.

    Returns:
        pd.Series: time-series of returns that accounts for stop-losses.
    """

    # Acquire only date indeces, rather than date-time indices
    dates = np.array([])
    for i in strategy_returns.index:
        dates = np.append(dates, i.date())

    # Traverse through daily returns
    for i in dates:

        # Convert date object into string that can be passed to pd.DataFrame.loc[]
        i = str(i)

        # Get all returns for given day (across all intraday returns)
        # Compute cumulative returns for given day
        tmp_cum_rets = strategy_returns.loc[i].cumsum()
        
        # If at any point cumulative reutrns hits stop loss target, flatten position and set daily return = stop-loss - market impact cost
        stop_loss_rets = tmp_cum_rets[tmp_cum_rets<stop_loss_target]
        stop_loss_triggered = len(stop_loss_rets) > 0

        if stop_loss_triggered:
            
            # Erase given day's returns
            # Set returns 0 after exit_date_time

            # In the future:
            # Keep prior returns before exit_date_time if need be
            # Set exit_date_time = -.01 - sum(prior returns)

            strategy_returns.loc[i] = 0

            # Get the date_time of when stop loss was triggered
            if len(stop_loss_rets) == 1:
                exit_date_time = i

            else:
                exit_date_time = tmp_cum_rets[tmp_cum_rets<stop_loss_target].index[0]

                if re_entry_eod == False:
                    # If positions are re-opened at the beginning of the next day, rather than EOD:
                    try:
                        # Get next trading day : str
                        next_day_index =str((pd.to_datetime(i) + pd.tseries.offsets.BDay(1)).date())
                        # Set next day's initial return to 0% since we liquidated the prior day's position
                        next_day_index = strategy_returns.loc[next_day_index].index[0]
                        strategy_returns.loc[next_day_index] = 0.0
                    except:
                        # May call exception if liquidation is on last day of returns data
                        print(f'Function portfolio_stop_loss: ensure {next_day_index} does not exist in strategy returns data')
                            
            # Realize losses at exit_date_time
            strategy_returns.loc[exit_date_time] = stop_loss_target - t_costs
            

    
    # After all stop losses have been realized, return strategy's return pd.Series
    return strategy_returns

def get_daily_stop_loss_returns(intraday_asset_returns: pd.DataFrame, mvo_wts: pd.DataFrame, stop_loss_target = -.01, re_entry_eod = False) -> pd.DataFrame:
    """ Generate daily strategy returns that incoporate a manager-specified daily strategy stop-loss target. This requires injestion of intraday data and
        MVO/strategy weights to determine stop-loss daily strategy returns. 

    Args:
        intraday_asset_returns (pd.DataFrame): _description_
        mvo_wts (pd.DataFrame): _description_
        stop_loss_target (float, optional): _description_. Defaults to -.005.
        re_entry_eod (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """


    # Convert daily MVO/strategy weights to intraday weights
    intraday_wts = get_intraday_weights(asset_returns = intraday_asset_returns, mvo_wts = mvo_wts)

    # Compute intraday returns
    intraday_strategy_returns = (intraday_asset_returns * intraday_wts).sum(1)
    # Compute intrday stop-loss returns
    intraday_strategy_returns_stop_loss = portfolio_stop_loss(strategy_returns = intraday_strategy_returns, stop_loss_target = stop_loss_target, re_entry_eod = re_entry_eod)
    
    # Convert intraday returns to daily returns - this makes it easier to compute strategy performance metrics (annualizing metrics)
    daily_strategy_returns_stop_loss = get_daily_rets_from_intraday(intraday_strategy_returns_stop_loss)

    return daily_strategy_returns_stop_loss 


# ------------------------------------------------------------------------- Performance Metrics -------------------------------------------------------------------------

# Get annualized volatility from daily returns
def vol(strategy_returns: pd.Series) -> float:
    """ Compute annualized volatility for any strategy/asset time-series of daily returns.

    Args:
        strategy_returns (pd.Series): time-series of daily returns.

    Returns:
        float: annualized volatility.
    """

    return strategy_returns.std() * 252 ** .5

# Get Sharpe Ratio of daily returns series
def sharpe_ratio(strategy_returns: pd.Series) -> float:
    """ Compute annualized Sharpe Ratio for any strategy or security time-series of daily returns.

    Args:
        strategy_returns (pd.Series): time-series of daily returns.

    Returns:
        float: annualized Sharpe Ratio.
    """
    
    return strategy_returns.mean() / strategy_returns.std() * 252 ** .5

def sortino_ratio(strategy_returns: pd.Series) -> float:
    """ Compute annualized Sortino Ratio for any strategy or security time-series of daily returns.
        This is the Sharpe Ratio, but with downside returns volatility.

    Args:
        strategy_returns (pd.Series): time-series of daily returns.

    Returns:
        float: annualized Sharpe Ratio.
    """
    
    return strategy_returns.mean() / strategy_returns[strategy_returns<0].std() * 252 ** .5

def omega_ratio(strategy_returns: pd.Series, required_return=0.07, required_return_annual = True, verbose = True) -> (pd.Series, float, float):

    """ Compute annualized Omega Ratio and Ultimate Omega Ratio for any strategy or security time-series of daily returns.
        This ratio captures tail-risk well by accounting for skew and kurtosis.
        
        The Ultimate Omega Ratio is the product of Omega ratios across 3 required_return thresholds: 
        "0", "required_return", and "required_return*2" return thresholds.

        For example, with an Omega ratio of 1.2, the given strategy outperforms its losses by a factor of 1.2 
        when accounting for leptokurtic and skewed returns behaviour.

        These ratios are highly robust and predictive risk-adjusted return metrics for out-of-sample performance.
        This is mainly due to their ability to account for all moments of a return distribution, and not depend on
        normally distributed returns like a Sharpe or Information Ratio.
    
    Args:
        strategy_returns (pd.Series): time-series of daily returns.
        required_returns (float, optional): minimum accepted daily return of the investor.Threshold over which to consider positive vs negative returns.
        required_return_annual (bool, optional): if True, required_returns argument is an annual return. Default is False, so required_returns is a daily returns
                                    
    Returns:
        pd.Series: 3 Annualized Omega Ratios
        float: 3 Annualized Omega Ratios Slope
        float: Ultimate Omega Ratio 
    
    Note:
        -------------------- Ultimate Omega Ratio --------------------
        Since the Ultimate Omega Ratio accounts for multiple required_return thresholds, it serves as a robust indicator of 
        out-of-sample performance when compared to competing strategies' Ultimate Omega Ratios.

        The steeper an omega curve, the less “risky” it is, in the sense that it has fewer extreme gains and losses. 
        This is because as you move the "required_return" from one number to another, if the returns are mostly clustered around the 
        median return, the drop-off is going to be pretty steep; if the returns are highly volatile and skewed, 
        moving the hurdle rate isn’t going to make such a big difference.

        See <https://en.wikipedia.org/wiki/Omega_ratio> and 
            <https://seekingalpha.com/article/4186730-ultimate-omega-best-risk-adjusted-performance-measure> 
            for more details.
    """

    # If annual reuired_return is passed, convert to daily required_return with: (1 + required_return) ** (1 / 252) - 1
    if required_return_annual == True:
        required_return = (1 + required_return) ** (1 / 252) - 1

    # ---------------------------------------- Step 1 - Omega 0% ----------------------------------------
    # Compute the Omega Ratio with a 0% threshold/required return (this is effectively the risk-reward ratio):
    omega_ratio_0 = get_risk_ratio(strategy_returns)
    
    # ---------------------------------------- Step 2 - Omega "required_return" ----------------------------------------
    # Compute Omega Ratio with a "required_return" threshold/required return
    returns_less_threshold = strategy_returns - required_return
    omega_gains = (returns_less_threshold[returns_less_threshold > 0.0]).sum()
    omgea_losses = np.abs((returns_less_threshold[returns_less_threshold < 0.0]).sum())
    # Standard Omega Ratio
    omega_ratio = omega_gains / omgea_losses

    # ---------------------------------------- Step 3 - Omega "2*required_return" ----------------------------------------
    # Compute Omega Ratio with a "2*required_return" threshold/required return
    returns_less_threshold_2 = strategy_returns - 2*required_return
    omega_gains_2 = (returns_less_threshold_2[returns_less_threshold_2 > 0.0]).sum()
    omgea_losses_2 = np.abs((returns_less_threshold_2[returns_less_threshold_2 < 0.0]).sum())
    # Omega Ratio of 2*required_return
    omega_ratio_2 = omega_gains_2 / omgea_losses_2
    
    # ---------------------------------------- Step 4 - Ultimate Omega + Slope ----------------------------------------
    # Compute the "Ultimate Omega Ratio" by computing the Omega ratio for a threshold of "0", "required_return', and 2 * "required_return"
    ultimate_omega_ratio = omega_ratio_0 * omega_ratio * omega_ratio_2

    # Aggregate all 3 Omega Ratios
    omega_summary = pd.Series({0 : omega_ratio_0, required_return : omega_ratio, required_return*2 : omega_ratio_2})

    # Compute slope of Omega Ratios
    omega_slope = stats.linregress(omega_summary.index, omega_summary).slope  

    if omega_gains > 0.0:
        if verbose == True:
            return omega_summary, omega_slope, ultimate_omega_ratio
        else:
            return omega_ratio
    else:
        return np.nan

# Get win rate of a strategy
def get_win_rate(strategy_returns: pd.Series, verbose=False) -> float:
    """ Compute win rate of a strategy.

    Args:
        strategy_returns (pd.Series): time series of daily returns.

    Returns:
        float: win rate as a decimal.
    """
    
    wins = len(strategy_returns[strategy_returns > 0])
    losses = len(strategy_returns[strategy_returns < 0])

    win_rate = wins / (wins + losses)

    required_rr_ratio = (1 / win_rate) - 1
    
    if verbose:
        print(f'Required Risk Ratio for Profit: {required_rr_ratio}')

    return  win_rate

# Get risk-reward ratio of a strategy
def get_risk_ratio(strategy_returns: pd.Series, verbose=False) -> float:
    """ Compute risk-reward ratio of a strategy. This can be utilized in conjunction
        with a strategy's win rate to ensure sufficient margins for profit.

        For example, a risk-reward ratio of 5:1 will require a 20% win rate for profit.
        Thus, if the win rate is significantly higher than 20%, we have sufficient margins 
        for a positive expected value (EV).

    Args:
        strategy_returns (pd.Series): time series of daily returns.

    Returns:
        float: risk-reward ratio as a decimal.
    """
    
    total_risk = np.abs(strategy_returns[strategy_returns < 0].sum())
    total_reward = strategy_returns[strategy_returns > 0].sum() 

    rr_ratio = total_reward / total_risk

    required_win_rate = 1 / (1+ rr_ratio)

    if verbose:
        print(f'Required Win Rate for Profit: {required_win_rate}')

    return rr_ratio

def alpha_regression(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.Series:
    """ Using daily returns, regress strategy returns on benchmark returns to compute strategy/portfolio alpha.

    Args:
        strategy_returns (pd.Series): time-series of strategy or portfolio daily returns.
        benchmark_returns (pd.Series): benchmark daily returns, normally S&P500, but could be a competing strategy as well.

    Returns:
        pd.Series: summary regression data relevant to strategy performance evaluation.
    """

    benchmark_returns = benchmark_returns.dropna()
    strategy_returns = strategy_returns.dropna()    

    indices = benchmark_returns.index.intersection(strategy_returns.index)

    benchmark_returns = benchmark_returns.loc[indices]
    strategy_returns = strategy_returns.loc[indices]

    # Add constant to benchmark returns
    benchmark_returns_const = sm.add_constant(benchmark_returns)

    # Construct OLS regression
    model = sm.OLS(strategy_returns, benchmark_returns_const)
    reg = model.fit()
    
    # ------------------------------ Store regression data in summary DataFrame ------------------------------

    reg_summary = pd.DataFrame({f'{strategy_returns.name} ~ {benchmark_returns.name}' : 
                                {
                                'Strategy Alpha' : reg.params[0]*252, # Annualize Alpha
                                'Alpha T-Stat' : reg.tvalues[0], 
                                'Strategy Beta' : reg.params[1], 
                                'Beta T-Stat' : reg.tvalues[1]
                                }
                            })

    reg_summary = reg_summary.reindex(index = ['Strategy Alpha', 'Alpha T-Stat', 'Strategy Beta', 'Beta T-Stat'])

    return reg_summary

# Get Performance Summary
def performance_summary(strategy_returns: pd.Series) -> pd.DataFrame:
    """ Generate pd.DataFrame of relevent strategy/portfolio performance data.

    Args:
        strategy_returns (pd.Series): time-series of strategy/portfolio daily returns.

    Returns:
        pd.DataFrame: 
    """

    performance_summary = pd.Series({'Sharpe Ratio' : sharpe_ratio(strategy_returns), 
                                    'Sortino Ratio' : sortino_ratio(strategy_returns), 
                                    'Omega Ratio' : omega_ratio(strategy_returns, verbose=False), 
                                    'Vol' : vol(strategy_returns),
                                    'RR Ratio' : get_risk_ratio(strategy_returns), 
                                    'Win Rate' : get_win_rate(strategy_returns),                           
                                    })

    return performance_summary

# ------------------------------------------------------------------------- Data Conversions -------------------------------------------------------------------------

# Scale strategy returns to a target volatility
def scale_vol(strategy_returns: pd.Series, target_vol = .10) -> pd.Series:
    """ Scale strategy returns to a target volatility.

    Args:
        strategy_returns (pd.Series): time-series of returns.
        target_vol (float, optional): targeted volatility for strategy. Defaults to .10.

    Returns:
        pd.Series: volatility scaled strategy returns
    """
    # Use vol_scalar to multiply strategy_returns by to realize target_vol
    vol_scalar = target_vol / vol(strategy_returns = strategy_returns)
    
    # Scale returns
    strategy_returns = strategy_returns * vol_scalar

    return strategy_returns

# Convert intraday returns to their respective daily returns
def get_daily_rets_from_intraday(strategy_returns: pd.Series) -> pd.Series:
    """ Takes intraday strategy returns, and converts them to daily returns.
        This conversion function is useful for analyzing the performance of a strategy or security 
        on an annualized basis. 

    Args:
        strategy_returns (pd.Series): intraday time series of strategy/security returns.
    
    Returns:
        pd.Series: converted daily time-series of strategy/security returns.
    """
    

    # Get the proper daily date range of returns
    daily_indices = pd.date_range(start=strategy_returns.index[0], end = strategy_returns.index[-1], freq='D')

    # Check if strategy_returns are already daily returns... this can be done by determining if there 
    # are more than 1 values per date (intrday) or only a single value (daily)
    if type(strategy_returns.loc[str(daily_indices[0].date())]) is not pd.Series:

        return strategy_returns

    else:

        # Initialize daily_returns dict to update as daily_indices is iterated through
        daily_returns = {}

        for date in daily_indices:

            # Get daily cumulative returns based on intraday returns
            # To access specified day's intraday returns, must refer to "date" as a string
            daily_cum_rets = strategy_returns.loc[str(date.date())].cumsum()

            if len(daily_cum_rets) > 0:
                # If cumulative returns is valid, set daily return to final cumulative return value
                daily_returns[str(date.date())] = daily_cum_rets.iloc[-1]
            else:
                pass # These are weekends
                # daily_returns[str(i.date())] = 0

        # Convert dict -> pd.Series of returns
        daily_returns = pd.Series(daily_returns)

        # Convert indices to proper format -> pd.DatetimeIndex
        daily_returns.index = pd.DatetimeIndex(daily_returns.index)

        return daily_returns
    
# Convert rebalancing weights (either daily or every n rebalance day) to intraday weights
def get_intraday_weights(asset_returns: pd.DataFrame, mvo_wts: pd.DataFrame) -> pd.DataFrame:
    """ Applies MVO or other targeted asset weights to intraday asset returns.

    Args:
        asset_returns (pd.DataFrame): historical intraday asset returns.
        mvo_wts (pd.DataFrame): targeted asset weights for given strategy, normally generated via Mean Variance Portfolio Optimization.

    Returns:
        pd.DataFrame: complete intraday pd.DataFrame of targeted asset weights
    """

    # Initialize intraday_wts pd.DataFrame
    intraday_wts = pd.DataFrame()
    intraday_wts.index = pd.to_datetime(asset_returns.index, utc = True)
    
    # Get MVO / other targeted asset weights
    # Must convert dates to account for time-zone via "utc = True" argument in pd.to_datetime function
    print(mvo_wts.index)
    mvo_wts.index = pd.to_datetime(mvo_wts.index, utc = True)

    # Apply MVO weights to intraday_wts pd.DataFrame
    intraday_wts = pd.concat([intraday_wts, mvo_wts], axis=1).ffill()

    # Get overlapping subset of mvo_wts by using the first index of asset_returns
    intraday_wts = intraday_wts.loc[asset_returns.index[0]:]

    return intraday_wts
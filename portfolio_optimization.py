'''
Library for portfolio optimization.

'''

from scipy.optimize import minimize as opt
from backtest_tools import risk_analysis as ra, portfolio_tools as pt, data_preprocessing as dp
from scipy.optimize import Bounds
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np



# ------------------------------------------------------------------------- Objective Functions -------------------------------------------------------------------------

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

def portfolio_dasr(w: pd.Series, returns: pd.DataFrame, neg = True) ->  float:
    """ Computes DASR of weighted portfolio.

    Args:
        betas (pd.Series): daily expected returns from normalized linear regression.
        squared_residuals (pd.DataFrame): squared error from OLS regression.
        w (pd.Series): portfolio weights.

    Returns:
        float: _description_
    """

    # Get weighted returns
    returns = (returns * w).sum(1)
    # Get weighted portfolio DASR
    portfolio_dasr = pt.drift_adjusted_sharpe_ratio(returns.dropna())

    if neg:
        # Return negative DASR for portfolio optimization
        return (-portfolio_dasr)
    
    # Return real DASR for performance analysis purposes
    return portfolio_dasr

def risk_parity_obj(w: pd.Series, cov: pd.DataFrame) -> float:
    """ Traditional risk parity objective function to minimize.
        Minimization ensures portfolio acheives equal variance contribution.       
 
    Args:
        w (pd.Series): portfolio weights.
        cov (pd.DataFrame): covariance matrix of historical portfolio contstituents' returns.

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
    weighted_absolute_risk = w.T.dot(cov) * w

    # Get each constituent's proportion risk contribution
    risk_contribution = weighted_absolute_risk / variance

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
    weighted_dollar_risk = n_units.T.dot(cov) * n_units
    # Get percent weighted risk contribution (percent contribution to variance)
    percent_risk_contribution = weighted_dollar_risk / variance

    # Check if the dollar risk contribution of current portfolio = equal_risk_contribution
    # Measure the absolute difference between current vs. equal risk contribution
    diff = np.abs((percent_risk_contribution - equal_risk_contribution)).sum()

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
    weighted_true_range_risk = n_units.T.dot(cov) * n_units
    
    # Get percent weighted risk contribution (percent contribution to variance)
    percent_risk_contribution = weighted_true_range_risk / portfolio_true_range_risk

    # Check if the dollar risk contribution of current portfolio = equal_risk_contribution
    # Measure the absolute difference between current vs. equal risk contribution
    diff = np.abs((percent_risk_contribution - equal_risk_contribution)).sum()

    return diff 

# ------------------------------------------------------------------------- Optimization Functions -------------------------------------------------------------------------

def mvo(hist_returns: pd.DataFrame, expected_returns: pd.DataFrame, long_only = False, vol_target = .01, max_position_weight = .2, net_exposure = 1, market_bias = 0, constrained=True, verbose=True):
    """ Constrained or unconstrained Mean Variance Optimization. This leverages convex optimization to identify local minima which serve to minimize an objective function.
        In the context of portfolio optimization, our objective function is the negative portfolio SR. 

    Args:
        hist_returns (pd.DataFrame): expanding historical returns of specified asset universe
        expected_returns (pd.DataFrame): expected returns across specified asset universe, normally computed via statistical model
        vol_target (float, optional): targeted ex-ante volatilty based on covariance matrix. Defaults to .10.

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

        if long_only: 
            bounds = Bounds(0, max_position_weight)
        else:
            # Set max allocation per security
            bounds = Bounds(-max_position_weight, max_position_weight)

        if constrained:

            constraints =  [# Target volatility
                            {"type": "eq", "fun": lambda w: np.sqrt(np.dot(np.dot(w.T, cov_matrix), w)) - vol_target},
                            
                            # Ensure market neutral portfolio (or alternatively specified market bias)
                            {"type": "eq", "fun": lambda w: np.sum(w) - market_bias},

                            # Target Leverage (Net Exposure)
                            {"type": "ineq", "fun": lambda w: np.sum(np.abs(w)) - (net_exposure - .01)}, # 0.99 <= weights.sum
                            {"type": "ineq", "fun": lambda w: (net_exposure + .01) - np.sum(np.abs(w))}, # 1.01 >= weights.sum
                            
                            # DO NOT DO THIS - Restrict optimizer too much {"type": "eq", "fun": lambda vols: 1 - np.sum(np.abs(vols))}, # 1.1 >= weights.sum
                            # 
                            #
                            # Eventually implement L/S Ratios 
                            # SUM(ABS(SHORTS)) = C * SUM(LONGS)
                            ]

            mvo_weights = pd.Series(opt(portfolio_sharpe_ratio, 
                    initial_guess,
                    args=(expected_returns, cov_matrix), 
                    method='SLSQP', 
                    bounds = bounds,
                    constraints=constraints)['x']
                    )
            
        elif vol_target is not None:
            
            constraints =  [# Target volatility
                            {"type": "eq", "fun": lambda w: np.sqrt(np.dot(np.dot(w.T, cov_matrix), w)) - vol_target},
                           ]

            mvo_weights = pd.Series(opt(portfolio_sharpe_ratio, 
                    initial_guess,
                    args=(expected_returns, cov_matrix), 
                    method='SLSQP', 
                    bounds = bounds,
                    constraints=constraints)['x']
                    )

        else:

            mvo_weights = pd.Series(opt(portfolio_sharpe_ratio, 
            initial_guess,
            args=(expected_returns, cov_matrix), 
            method='SLSQP')['x']
            )             

        # Assign weights to assets
        mvo_weights.index = vols.index


        # Compute absolute value of all vols, sum them together, and scale vols by 1 / (sum(abs(vols)))
        # This ensures that the new sum of absolute values of vols = 1 (i.e., generates true weights or fractions of the portfolio to allocate per investment)
        # mvo_weights = mvo_weights / np.sum(np.abs(mvo_weights))
        
        # ---------------------------------- Print relevant allocation information ----------------------------------
        if verbose:
            print('Target Vol: ')
            print(np.sqrt(np.dot(np.dot(mvo_weights.T, cov_matrix), mvo_weights)))
            
            ls_ratio = np.abs(mvo_weights[mvo_weights>0].sum() / mvo_weights[mvo_weights<0].sum())
            print(f'Long-Short Ratio: {ls_ratio}')
            print(f'Leverage: {mvo_weights.abs().sum()}')
            print(f'Sum of Vol Weights: {mvo_weights.sum().round(4)}')
            mvo_sr = portfolio_sharpe_ratio(mvo_weights, expected_returns, cov_matrix, neg=False)
            print(f'Target Portfolio Sharpe Ratio: {mvo_sr}')
        
        return mvo_weights
    return

def unconstrained_mvo(hist_returns: pd.DataFrame, expected_returns: pd.Series):
    """ Implements MVO closed-form solution for maximizing Sharpe Ratio.

    Args:
        hist_returns (pd.DataFrame): pd.DataFrame of historical returns.
        expected_returns (pd.Series): series of expected returns.

    Returns:
        _type_: _description_
    """
    # Get E[SR] and Correlation Matrix
    expected_sr = hist_returns.mean()/hist_returns.std()*252**.5 # hist_returns.mean()
    inverse_corr = np.linalg.inv(hist_returns.corr()) # np.linalg.inv(hist_returns.cov()).round(4) 

    # Get MVO Vol Weights and Convert them to Standard Portfolio Weights
    numerator = np.dot(inverse_corr, expected_sr)
    denominator = np.sum(numerator)
    mvo_weights = numerator / denominator
    mvo_weights = pd.Series(mvo_weights, index=hist_returns.columns)

    # ---------------------------------- Print relevant allocation information ----------------------------------
    cov_matrix = hist_returns.cov()
    print('Target Vol: ')
    print(np.sqrt(np.dot(np.dot(mvo_weights.T, cov_matrix), mvo_weights)))
    
    ls_ratio = np.abs(mvo_weights[mvo_weights>0].sum() / mvo_weights[mvo_weights<0].sum())
    print(f'Long-Short Ratio: {ls_ratio}')
    print(f'Leverage: {mvo_weights.abs().sum()}')
    print(f'Sum of Vol Weights: {mvo_weights.sum().round(4)}') 
    mvo_sr = portfolio_sharpe_ratio(mvo_weights, expected_returns, cov_matrix, neg=False)
    print(f'Target Portfolio Sharpe Ratio: {mvo_sr}')   
    
    return mvo_weights

def risk_parity(returns: pd.DataFrame, type='Variance') -> pd.Series:
    """ Generalized Risk Parity portfolio construction algorithm to 
        get equal risk contribution portfolio weights across various defintions
        of risk. 

        CURRENT RISK IMPLEMENTATIONS: 
        - Variance 

    Args:
        returns (pd.DataFrame): portfolio constituents' historical returns
        type (str, optional): defintion of risk to neutralize. Defaults to 'Variance'.

    Returns:
        pd.Series: risk parity portfolio weights
    """

    # Determine risk definition
    if type=='Variance':
        objective_function = risk_parity_obj

    n = len(returns.columns)

    cov = returns.cov()  

    initial_guess = pd.Series(np.array([1/n] * n), index=returns.columns)

    constraints =   [# Portfolio weights sum to 100%
                    {"type": "eq", "fun": lambda w: w.sum() - 1}
                    ]

    # Get risk parity weights
    w = opt(objective_function, 
                initial_guess, 
                args=(cov), 
                method='SLSQP',
                constraints=constraints)['x']

    w = pd.Series(w, index=cov.index)

    return w

def dollar_risk_parity(prices: pd.DataFrame, target_risk = 0.001, portfoio_value = 100000, long_only=False) -> pd.Series:
    """ Inspired by CTA and Trend Follwers' risk management practices, the Dollar Risk Parity optimization function
        targets "target_risk" percent risk per position (equal risk contribution) which represents 1 SD of the 
        underlying instrument's price movement. Here, the covariance and variance of each asset defines its risk to account for
        correlation structures and mitiage over exposure to a single risk factor.
               
        Positions represent how many shares to purchase for a 1SD move in the underlying instrument 
        to represent "target_risk" percent loss in "portfolio_value". This follows the risk management practices of 
        select portfolio managers that target equal risk allocation to each trade/position, but leverages stop-losses 
        (e.g., 1SD stop) to limit exogenous risk exposure.

        This follows the intuition behind traditional equal variance contribution risk parity, but defines risk as dollar value
        at risk (e.g., stop loss set at 1SD) instead of purely variance. 

    Args:
        prices (pd.DataFrame): _description_
        target_risk (float, optional): _description_. Defaults to 0.005.
        portfoio_value (int, optional): _description_. Defaults to 100000.

    Returns:
        pd.Series: _description_
    """

    # Determine risk definition
    objective_function = dollar_risk_parity_obj

    n = len(prices.columns)

    # Covariance of prices
    cov = prices.cov() 

    initial_guess = pd.Series(np.array([1/n] * n), index=prices.columns)

    constraints =   [# Ensure notional exposure < portfolio_value (i.e., no leverage)
                    {"type": "ineq", "fun": lambda n_units: portfoio_value - (n_units*prices.iloc[-1]).sum()}
                    ]

    # Long-Only or L/S
    if long_only:
        bounds = Bounds(0, np.inf)
        # bounds = Bounds(0, 10000000000)
    else:
        bounds = Bounds(-np.inf, np.inf)
        # bounds = Bounds(-10000000000, 10000000000)
    
    # Get dollar risk parity weights
    n_units = opt(objective_function, 
            initial_guess, 
            bounds=bounds,
            args=(cov),
            method='SLSQP',
            constraints=constraints)['x']

    n_units = pd.Series(n_units, index=cov.index)

    ex_ante_dollar_vol = np.sqrt(n_units.T.dot(cov).dot(n_units))
    print(f"Target Portfolio Risk: {portfoio_value*n*target_risk}")
    
    risk_scalar = portfoio_value*n*target_risk / ex_ante_dollar_vol

    n_units *= risk_scalar
    ex_ante_scaled_dollar_vol = np.sqrt(n_units.T.dot(cov).dot(n_units))
    print(f"Ex-Ante Portfolio Risk: {ex_ante_scaled_dollar_vol}")
    print(f"Ex-Ante Dollar Risk Contributions: \n{(n_units.T.dot(cov) *  n_units)**.5}" )

    return n_units


def atr_risk_parity(prices: pd.DataFrame, true_ranges: pd.DataFrame, target_risk = 0.001, portfoio_value = 100000, long_only=False) -> pd.Series:
    """ Inspired by CTA and Trend Follwers' risk management practices, the Dollar Risk Parity optimization function
        targets "target_risk" percent risk per position (equal risk contribution) which represents 1 SD of the 
        underlying instrument's price movement. Here, the covariance and variance of each asset defines its risk to account for
        correlation structures and mitiage over exposure to a single risk factor.
               
        Positions represent how many shares to purchase for a 1SD move in the underlying instrument 
        to represent "target_risk" percent loss in "portfolio_value". This follows the risk management practices of 
        select portfolio managers that target equal risk allocation to each trade/position, but leverages stop-losses 
        (e.g., 1SD stop) to limit exogenous risk exposure.

        This follows the intuition behind traditional equal variance contribution risk parity, but defines risk as dollar value
        at risk (e.g., stop loss set at 1SD) instead of purely variance. 

    Args:
        prices (pd.DataFrame): _description_
        target_risk (float, optional): _description_. Defaults to 0.005.
        portfoio_value (int, optional): _description_. Defaults to 100000.

    Returns:
        pd.Series: _description_
    """

    # Determine risk definition
    objective_function = atr_risk_parity_obj

    n = len(prices.columns)

    # Covariance of prices
    cov = dp.true_range_covariance(true_ranges, lookback_window=20) 

    initial_guess = pd.Series(np.array([1/n] * n), index=prices.columns)

    constraints =   [# Notional exposure < portfolio_value (i.e., no leverage)
                    {"type": "ineq", "fun": lambda n_units: portfoio_value - prices.iloc[-1].dot(n_units)},
                    # Target Risk Level - Doesn't Work - Scale Risk After Opt Instead
                    # {"type": "eq", "fun": lambda n_units: np.sqrt(n_units.T.dot(cov).dot(n_units)) - portfoio_value*n*target_risk}
                    ]

    # Long-Only or L/S
    if long_only:
        bounds = Bounds(0, np.inf)
        # bounds = Bounds(0, 10000000000)
    else:
        bounds = Bounds(-np.inf, np.inf)
        # bounds = Bounds(-10000000000, 10000000000)
    
    # Get dollar risk parity weights
    n_units = opt(objective_function, 
            initial_guess, 
            bounds=bounds,
            args=(cov),
            method='SLSQP',
            constraints=constraints)['x']

    n_units = pd.Series(n_units, index=cov.index)

    # Target Portfolio Risk
    ex_ante_true_range_risk = np.sqrt(n_units.T.dot(cov).dot(n_units)) 
    risk_scalar = portfoio_value*n*target_risk / ex_ante_true_range_risk
    n_units *= risk_scalar

    # Control for Leverage -- alternative is to impose this in the convex optimization constraints
    leverage_scalar = portfoio_value / prices.iloc[-1].dot(n_units)
    n_units *= leverage_scalar
    
    ex_ante_true_range_risk = np.sqrt(n_units.T.dot(cov).dot(n_units))
    print(f"Target Portfolio Risk: {portfoio_value*n*target_risk}")
    print(f"Ex-Ante True Range Dollar Risk: {ex_ante_true_range_risk}")
    print(f"Ex-Ante True Range Risk Contributions: \n{(n_units.T.dot(cov) *  n_units)**.5}" )

    return n_units

def dpo(returns: pd.DataFrame, long_only=False, constrained=True, max_position_weight=1, vol_target=None) -> pd.Series:
    """ Executes constrained convex portfolio optimization to generate optimal
        DASR asset weights.

    Args:
        returns (pd.DataFrame): _description_
        long_only (bool, optional): _description_. Defaults to False.
        constrained (bool, optional): _description_. Defaults to True.
        max_position_weight (int, optional): _description_. Defaults to 1.
        vol_target (_type_, optional): _description_. Defaults to None.

    Returns:
        pd.Series: _description_
    """

    n = len(returns.columns)

    # Initial guess is naive 1/n portfolio
    w = np.array([1 / n] * n)

    # Declare constraints list
    constraints =  []

    # Set constraints (e.g., leverage, vol target, max sizing)
    if constrained:

        # Max position size
        if long_only:
            # Long only constraint
            bounds = Bounds(0, max_position_weight)
        else:
            # L/S constraint
            bounds = Bounds(-max_position_weight, max_position_weight)

        # No Leverage Constraint
        constraints.append({"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1})

        # Volatility Targeting
        if vol_target is not None:           
            constraints.append({"type": "eq", "fun": lambda w: np.sqrt(np.dot(np.dot(w.T, returns.cov()), w)) - vol_target})
    else:
        # Is not implementable nor optimal
        bounds = Bounds(None, None)

        constraints =  []
    
    # Get optimized weights
    w = pd.Series(opt(portfolio_dasr, 
                            w,
                            args=(returns), 
                            method='SLSQP',
                            bounds = bounds,
                            constraints=constraints)['x'],
                index=returns.columns
                )
    
    return w

def multistrategy_portfolio_optimization(multistrategy_portfolio: pd.DataFrame, rebal_freq: int, lookback_window: int, optimization = ['MVO', 'DPO'], vol_target=None, max_position_weight=.2, constrained=True):
    """ Executes multistrategy portfolio optimization by leveraging either one of two different optimization algorithms: 
        - MVO
        - Drift Adjusted Portfolio Optimization

    Args:
        multistrategy_portfolio (pd.DataFrame): strategy returns.
        rebal_freq (int):
        lookback_window (int): 
        optimiztion (list, optional): type of optimization. Defaults to ['MVO', 'DPO'].
        vol_target (float, optional): Defaults to .01.
        max_position_weight (float, optional): Defaults to .5.
        constrained (boolean, optiona;): 
    """

    w = {}

    if optimization == 'MVO':
        for date in multistrategy_portfolio.index[::rebal_freq]:
        
            w[date] = mvo(hist_returns=multistrategy_portfolio.loc[:date], expected_returns=multistrategy_portfolio.loc[:date].tail(lookback_window).mean(), vol_target=vol_target, max_position_weight=max_position_weight, constrained=constrained, verbose=False, long_only=True)
    
    elif optimization == 'DPO':
        for date in multistrategy_portfolio.index[::rebal_freq]:
            w[date] = dpo(returns=multistrategy_portfolio.loc[:date].tail(lookback_window), long_only=True, constrained=constrained, max_position_weight=max_position_weight, vol_target=vol_target)

    # Convert Hash Table to DataFrame
    indices_df = pd.DataFrame(index=multistrategy_portfolio.index)
    w = pd.concat([indices_df, pd.DataFrame(w).T], axis=1).ffill().dropna()

    # Get strategy returns
    multistrategy_portfolio_optimized_returns = (multistrategy_portfolio*w.shift(2)).sum(1).dropna()

    return multistrategy_portfolio_optimized_returns, w

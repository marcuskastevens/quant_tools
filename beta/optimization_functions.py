'''
Library for portfolio optimization functions.

Supported Optimization Algorithms:
1) Unconstrained Max Sharpe Ratio (Closed-Form Lagrangian) 
2) Max Sharpe Ratio (MVO)
3) Risk Parity (Equal Variance Contribution)
4) Dollar Risk Parity (Equal Dollar Contribution of Variance Risk)
5) ATR Risk Parity (Equal Dollar Contribution of "True Range" Risk)

Future Optimization Algorithms:
1) Min Variance
2) Max Ucler Index / Martin Ratio
3) Max Calmar Ratio
4) HRP 
5) NCO
6) Min Entropic VaR / Minimum Kurtosis Portfolios
7) Max Sortino 
'''


from scipy.optimize import minimize as opt
from quant_tools import risk_analysis as ra, performance_analysis as pt, data_preprocessing as dp
from quant_tools.beta import obj_functions
from scipy.optimize import Bounds
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------- Optimization Functions -------------------------------------------------------------------------

def unconstrained_max_sharpe_mvo(hist_returns: pd.DataFrame, expected_returns: pd.Series, verbose: bool = False) -> pd.Series:
    """ Implements MVO closed-form solution for maximizing Sharpe Ratio.

    Args:
        hist_returns (pd.DataFrame): historical returns matrix.
        expected_returns (pd.Series): expected returns vector.
        verbose (bool): if true, print relevant optimization information.

    Returns:
        pd.Series: optimized weights vector.
    """
    # Get E[SR] and Correlation Matrix
    expected_sr = hist_returns.mean()/hist_returns.std()*252**.5 # hist_returns.mean()
    inverse_corr = np.linalg.inv(hist_returns.corr()) # np.linalg.inv(hist_returns.cov()).round(4) 

    # Get MVO Vol Weights and Convert them to Standard Portfolio Weights
    numerator = np.dot(inverse_corr, expected_sr)
    denominator = np.sum(numerator)
    w = numerator / denominator
    w = pd.Series(w, index=hist_returns.columns)

    # Print relevant allocation information
    if verbose:
        cov = hist_returns.cov()
        print('Target Vol: ')
        print(np.sqrt(np.dot(np.dot(w.T, cov), w)))
        
        ls_ratio = np.abs(w[w>0].sum() / w[w<0].sum())
        print(f'Long-Short Ratio: {ls_ratio}')
        print(f'Leverage: {w.abs().sum()}')
        print(f'Sum of Vol Weights: {w.sum().round(4)}') 
        mvo_sr = obj_functions.sharpe_ratio_obj(w, expected_returns, cov, neg=False)
        print(f'Target Portfolio Sharpe Ratio: {mvo_sr}')   
    
    return w


def max_sharpe_mvo(hist_returns: pd.DataFrame, expected_returns: pd.DataFrame, long_only = False, vol_target = .01, max_position_weight = .2, net_exposure = 1, market_bias = 0, constrained=True, verbose=False) -> pd.Series:
    """ Constrained or unconstrained Mean Variance Optimization. This leverages convex optimization to identify local minima which serve to minimize an objective function.
        In the context of portfolio optimization, our objective function is the negative portfolio SR. 

    Args:
        hist_returns (pd.DataFrame): expanding historical returns of specified asset universe
        expected_returns (pd.DataFrame): expected returns across specified asset universe, normally computed via statistical model
        vol_target (float, optional): targeted ex-ante volatilty based on covariance matrix. Defaults to .10.

    Returns:
        pd.Series: optimized weights vector.
    """

    # Match tickers across expected returns and historical returns
    expected_returns.dropna(inplace=True)
    hist_returns = hist_returns.loc[:, expected_returns.index]

    cov = hist_returns.cov()
    
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
                            {"type": "eq", "fun": lambda w: np.sqrt(np.dot(np.dot(w.T, cov), w)) - vol_target},
                            
                            # Ensure dollar neutral portfolio (or alternatively specified market bias)
                            {"type": "eq", "fun": lambda w: np.sum(w) - market_bias},

                            # Target Leverage (Net Exposure)
                            {"type": "ineq", "fun": lambda w: np.sum(np.abs(w)) - (net_exposure - .01)}, # 0.99 <= weights.sum
                            {"type": "ineq", "fun": lambda w: (net_exposure + .01) - np.sum(np.abs(w))}, # 1.01 >= weights.sum

                            ]

            w = pd.Series(opt(obj_functions.sharpe_ratio_obj, 
                    initial_guess,
                    args=(expected_returns, cov), 
                    method='SLSQP', 
                    bounds = bounds,
                    constraints=constraints)['x']
                    )
            
        elif vol_target is not None:
            
            constraints =  [# Target volatility
                            {"type": "eq", "fun": lambda w: np.sqrt(np.dot(np.dot(w.T, cov), w)) - vol_target},
                           ]

            w = pd.Series(opt(obj_functions.sharpe_ratio_obj, 
                    initial_guess,
                    args=(expected_returns, cov), 
                    method='SLSQP', 
                    bounds = bounds,
                    constraints=constraints)['x']
                    )

        else:

            w = pd.Series(opt(obj_functions.sharpe_ratio_obj, 
            initial_guess,
            args=(expected_returns, cov), 
            method='SLSQP')['x']
            )             

        # Assign weights to position labels
        w.index = hist_returns.columns
       
        # Print relevant allocation information
        if verbose:
            print('Target Vol: ')
            print(np.sqrt(np.dot(np.dot(w.T, cov), w)))
            
            ls_ratio = np.abs(w[w>0].sum() / w[w<0].sum())
            print(f'Long-Short Ratio: {ls_ratio}')
            print(f'Leverage: {w.abs().sum()}')
            print(f'Sum of Vol Weights: {w.sum().round(4)}')
            mvo_sr = obj_functions.sharpe_ratio_obj(w, expected_returns, cov, neg=False)
            print(f'Target Portfolio Sharpe Ratio: {mvo_sr}')
        
        return w
    return

# # Risk Averse MVO
# def risk_averse_mvo(hist_returns: pd.DataFrame, expected_returns: pd.DataFrame, long_only = False, vol_target = .01, max_position_weight = .2, net_exposure = 1, market_bias = 0, constrained=True, verbose=False) -> pd.Series:

# # Risk & Turnover Averse MVO
# def risk_turnover_averse_mvo(hist_returns: pd.DataFrame, expected_returns: pd.DataFrame, long_only = False, vol_target = .01, max_position_weight = .2, net_exposure = 1, market_bias = 0, constrained=True, verbose=False) -> pd.Series:

def risk_parity(hist_returns: pd.DataFrame, long_only=False) -> pd.Series:
    """ Generalized Risk Parity portfolio construction algorithm to 
        get equal risk contribution (variance) portfolio weights.

    Args:
        returns (pd.DataFrame): portfolio constituents' historical returns.
        type (str, optional): defintion of risk to neutralize. Defaults to 'Variance'.

    Returns:
        pd.Series: risk parity portfolio weights.
    """

    n = len(hist_returns.columns)

    cov = hist_returns.cov()  

    initial_guess = pd.Series(np.array([1/n] * n), index=hist_returns.columns)

    # Long-Only or L/S
    if long_only:
        bounds = Bounds(0, np.inf)
    else:
        bounds = Bounds(-np.inf, np.inf)

    constraints =   [# Portfolio weights sum to 100%
                    {"type": "eq", "fun": lambda w: w.sum() - 1}
                    ]

    # Get risk parity weights
    w = opt(obj_functions.risk_parity_obj, 
                initial_guess, 
                args=(cov), 
                method='SLSQP',
                bounds=bounds,
                constraints=constraints)['x']

    w = pd.Series(w, index=cov.index)

    return w

def dollar_risk_parity(prices: pd.DataFrame, target_risk = 0.001, portfoio_value = 100000, long_only=False, verbose=False) -> pd.Series:
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
    else:
        bounds = Bounds(-np.inf, np.inf)

    # Get dollar risk parity weights
    n_units = opt(obj_functions.dollar_risk_parity_obj, 
            initial_guess, 
            bounds=bounds,
            args=(cov),
            method='SLSQP',
            constraints=constraints)['x']

    # Assign units to position labels
    n_units = pd.Series(n_units, index=cov.index)

    # Compute target portfolio risk (dollar risk at 1SD move)
    target_portfolio_risk = portfoio_value*n*target_risk

    # Get ex-ante dollar risk at 1SD
    ex_ante_dollar_vol = np.sqrt(n_units.T.dot(cov).dot(n_units))
    
    # Compute risk scalar to ensure target risk is acheived
    risk_scalar = target_portfolio_risk / ex_ante_dollar_vol

    # Multiply optimized positions by risk scalar to target vol
    n_units *= risk_scalar

    # Re-compute ex-ante dollar risk at 1SD
    ex_ante_scaled_dollar_vol = np.sqrt(n_units.T.dot(cov).dot(n_units))

    if verbose:
        print(f"Target Portfolio Risk: {target_portfolio_risk}")
        print(f"Ex-Ante Portfolio Risk: {ex_ante_scaled_dollar_vol}")
        print(f"Ex-Ante Dollar Risk (1SD) Contributions: \n{(n_units.T.dot(cov) * n_units)**.5}" )

    return n_units


def atr_risk_parity(prices: pd.DataFrame, true_ranges: pd.DataFrame, lookback_window=20, target_risk = 0.001, portfoio_value = 100000, long_only=False, verbose=False) -> pd.Series:
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

    n = len(prices.columns)

    # Covariance of prices
    cov = dp.true_range_covariance(true_ranges, lookback_window=lookback_window)

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
    n_units = opt(obj_functions.atr_risk_parity_obj, 
            initial_guess, 
            bounds=bounds,
            args=(cov),
            method='SLSQP',
            constraints=constraints)['x']

     # Assign units to position labels
    n_units = pd.Series(n_units, index=cov.index)

    # Compute target portfolio risk (dollar risk at 1SD move)
    target_portfolio_risk = portfoio_value*n*target_risk

    # Get ex-ante dollar risk at 1SD
    ex_ante_dollar_vol = np.sqrt(n_units.T.dot(cov).dot(n_units))
    
    # Compute risk scalar to ensure target risk is acheived
    risk_scalar = target_portfolio_risk / ex_ante_dollar_vol

    # Multiply optimized positions by risk scalar to target vol
    n_units *= risk_scalar

    # Control for Leverage - alternative is to impose this in the convex optimization constraints
    if prices.iloc[-1].dot(n_units) > portfoio_value:
        print('Leverage Constraint Used')
        leverage_scalar = portfoio_value / prices.iloc[-1].dot(n_units)
        n_units *= leverage_scalar
    
    # Re-compute ex-ante dollar risk at 1SD
    ex_ante_true_range_risk = np.sqrt(n_units.T.dot(cov).dot(n_units))

    if verbose: 
        print(f"Target Portfolio Risk: {portfoio_value*n*target_risk}")
        print(f"Ex-Ante True Range Dollar Risk: {ex_ante_true_range_risk}")
        print(f"Ex-Ante True Range Risk Contributions: \n{(n_units.T.dot(cov) *  n_units)**.5}")

    return n_units

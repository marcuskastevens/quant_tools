from scipy.stats import kurtosis, skew, t
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------- Performance Metrics -------------------------------------------------------------------------

# Compute N Largest Drawdowns
def get_max_n_drawdowns(returns, N = 10):
    """ Generate series of drawdowns and plot max N drawdowns.

    Args:
        returns (_type_): _description_
        N (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """

    # 100 serves as initial portfolio value... can be any number
    cum_returns = returns.cumsum()
    cum_returns = (1+cum_returns)
    
    # Initialize a dict to store drawdown data
    drawdowns = dict()#pd.Series()

    for i, args in enumerate(cum_returns.items()):

        # Exctract args
        date, ret = args
        
        # Initialize max & min value which will be dynamically updated as we traverse through dates
        if i == 0:
            peak = ret
            trough = ret
            
        # Break out of drawdown
        if ret > peak and trough is not peak:
            
            # Update drawdowns dict and store drawdown's start and end dates
            index = f'{start_date} -- {date.date()}'
            drawdowns[index] = np.round(trough/peak-1, 4)

            # Reset variables
            trough = ret
            peak = ret  
                

        # Out of drawdown, new peak
        elif ret > peak:

            # If new peak, we set peak & trough equal since we're not in a drawdown
            peak = ret
            trough = ret
                    
        # In drawdown, new trough
        elif ret < trough: 
            
            # If this is the beginning of a new drawdown, update start_date
            if peak == trough:
                start_date = date.date()

            trough = ret

    # Change idices from datetimeindex to date
    #drawdowns.index = drawdowns.index.date
    drawdowns = pd.Series(drawdowns)
    get_n_max_drawdowns = drawdowns.nsmallest(N)

    # Plot N largest drawdowns
    get_n_max_drawdowns.plot.bar()
    
    return get_n_max_drawdowns

# Compute Time-Series of Drawdowns
def get_drawdowns(returns):
    """ Generate series of drawdowns and plot max N drawdowns.

    Args:
        returns (_type_): _description_
        N (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """

    # Initialize drawdowns dictionary
    drawdowns = dict()

    # Set initial high portfolio value
    prev_high = 0

    cumulative_returns = returns.cumsum() + 1

    # Loop through each day's cumulative return
    for date, cum_ret in cumulative_returns.items():
        
        # Update all-time high 
        prev_high = max(prev_high, cum_ret)

        # Compute current drawdown... will be negative if in a drawdown, otherwise, it will be zero
        dd = (cum_ret - prev_high) / prev_high

        # Update drawdowns
        drawdowns[date.date()] = dd

    drawdowns = pd.Series(drawdowns)
    return drawdowns

    
# ------------------------------------------------------------------------- Simulations -------------------------------------------------------------------------

# Var & CVar
def value_at_risk(returns, percentile = 5):
    """ Compute Value at Risk & Conditional Value at Risk from simulation returns

    Args:
        returns (pd.DataFrame): DataFrame of simulated returns from a Monte Carlo simulation.
        percentile (int): Var & CVar confidence level. Defaults to 5th percentile for 95% confidence.
    """
    if type(returns) == pd.DataFrame:
        # Compute daily Var & CVar at "percentile" confidence interval
        sim_daily_rets = returns.values.tolist()[0]
        Var = np.percentile(sim_daily_rets, percentile)
        # Convert all daily returns to np.array
        tmp_CVar_array = np.array(sim_daily_rets)
        # Isolate returns < Var
        tmp_CVar_array = np.where(tmp_CVar_array < Var, tmp_CVar_array, np.nan)
        # Delete NaNs and compute mean ~ CVar
        CVar = tmp_CVar_array[~np.isnan(tmp_CVar_array)].mean()

    elif type(returns) == pd.Series:
        # Compute daily Var & CVar at "percentile" confidence interval
        sim_daily_rets = returns.values.tolist()
        Var = np.percentile(sim_daily_rets, percentile)
        # Convert all daily returns to np.array
        tmp_CVar_array = np.array(sim_daily_rets)
        # Isolate returns < Var
        tmp_CVar_array = np.where(tmp_CVar_array < Var, tmp_CVar_array, np.nan)
        # Delete NaNs and compute mean ~ CVar
        CVar = tmp_CVar_array[~np.isnan(tmp_CVar_array)].mean()

    return Var, CVar


# Monte Carlo Simulations (Var, CVar)
def monte_carlo_Var (returns, days = 10, N = 1000, percentile = 5, use_students_t = False, plot=False):
    """ Conduct Monte Carlo simulations on a given time-series of strategy returns data. This will 
        enable the researcher to acquire relevant risk metrics and provide insight into the distribution of returns. 

    Args:
        returns (pd.Series): Series of returns data.
        days (int, optional): Duration of Monte-Carlo simulation to be plotted. Defaults to 10.
        N (int, optional): Number of simulations. Defaults to 1000.
        percentile (int, optional): Var & CVar condifence level. Defaults to 5th percentile for 95% confidence.
        use_students_t (bool, optional): Determines if the simulation will utilize both normal and student's t distributions. Defaults to False.
        plot (bool, optional): Determines if simulations will be plotted. Defaults to False.
        

    Returns:
        _type_: _description_
    """

    # Plot Empirical Returns Distribution
    plt.hist(np.sort(returns), 50)
    
    # Get daily returns' mean, std, kurtosis, skew, & df
    mu = returns.mean()
    sigma = returns.std()
    strat_kurtosis = kurtosis(returns, fisher=False)
    strat_skew = skew(returns)
    # Compute Student's T Distribution's "Degrees of Freedom" = 6/kurt + 4
    df = 6 / strat_kurtosis + 4

    # Nomral Distribution Monte Carlo Simulation on N iterations
    mc_norm = pd.DataFrame(np.random.normal(mu, sigma, (days, N)))
    # Compute Var & CVar
    norm_Var, norm_CVar = value_at_risk(mc_norm, percentile=percentile)
    full_sample_Var, full_sample_CVar = value_at_risk(mc_norm.cumsum().iloc[-1], percentile)
    
    if use_students_t == True:
        mc_t_standard = pd.DataFrame(np.array(t.rvs(df, size=(days, N)))) #1 Compute Student's T MC
        t_Var, t_CVar = value_at_risk(mc_t_standard, percentile=percentile)   
        print(f'T-Var: {t_Var}... T-CVar: {t_CVar}')
        # Undo Z-Score: 
        # mc_t = mc_t_standard/np.sqrt(df/(df-2)) # 2
        # mc_t = mc_t * sigma # 3
        # mc_t = mc_t + mu # 4
        # t_plot = plt.hist(np.sort(mc_t), 50)
        
    # Plot Normal Distribtuion MC result
    # norm_plot = plt.hist(mc_norm, 50)
    # plt.show()
    if plot == True:
        mc_norm_cumsum = mc_norm.cumsum().plot(title='Normal Distribution MC Sim', legend=False)
        plt.show()

        # Plot Student's T MC Results
        if use_students_t == True:
            mc_t_standard_cumsum = mc_t_standard.cumsum().head(252).plot(title='Student T Distribution MC Sim', legend=False)
            plt.show()


    return norm_Var, norm_CVar, full_sample_Var, full_sample_CVar # mu, sigma, strat_kurtosis, df, mc_norm,
    

# Vol limit & conditional vol limit (EV of Vol)
def vol_simulations(returns, target_vol = .10, lookback_period = 20, percentile = 99, N = 10000):
    """ Simulates volatility conditions and determines what threshold of volatility is abnomral.

    Args:
        returns (pd.Series): Series of returns data.
        target_vol (float, optional): Targeted annualized volatility used for scaling purposes. Defaults to .10.
        lookback_period (int, optional): Lookback period to compute trailing volatility. Defaults to 20 or 1-month.
        percentile (int, optional): Vol condifence level. Defaults to 5th percentile for 95% confidence.
    """

    # Compute vol scalar to ensure targeted vol
    vol_scalar = target_vol / (returns.std()*252**.5)
    scaled_returns = vol_scalar * returns

    # Compute trailing "lookback_period" annualized vol
    trailing_vol = (scaled_returns.rolling(lookback_period).std() * (252/lookback_period) **.5).dropna()

    # Compute empirical conditional vol limit and vol limit at 99th percentile (EV of Vol > Vol Limit at 99th percentile)
    empirical_vol_limit = np.percentile(trailing_vol, percentile)
    empirical_CVol_limit = np.where(np.array(trailing_vol) > empirical_vol_limit, np.array(trailing_vol), np.nan) 
    empirical_CVol_limit = empirical_CVol_limit[~np.isnan(empirical_CVol_limit)].mean()
    
    plt.hist(trailing_vol, 50)
    plt.show()
    
    # Simulate 10,000 samples using normal distribution (could be changed in the future to a more heterskedastic, platykurtic distribution)
    mu = trailing_vol.mean()
    sigma = trailing_vol.std()
    vol_sim = np.random.normal(mu, sigma, size=10000)
    # Ensure vols are not negative, if so, just set as mean vol
    vol_sim = np.where(vol_sim < 0, mu, vol_sim)

    # Compute simulated conditional vol limit and vol limit at 99th percentile (EV of Vol > Vol Limit at 99th percentile)
    sim_vol_limit = np.percentile(vol_sim, percentile)
    sim_CVol_limit = np.where(vol_sim > sim_vol_limit, vol_sim, np.nan) 
    sim_CVol_limit = sim_CVol_limit[~np.isnan(sim_CVol_limit)].mean()

    plt.hist(pd.Series(vol_sim), 50)
    plt.show()

    return sim_vol_limit, sim_CVol_limit, empirical_vol_limit, empirical_CVol_limit
from scipy.stats import norm, kurtosis, skew, t
import quant_tools.performance_analysis as pt
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ------------------------------------------------------------------------- Performance Metrics -------------------------------------------------------------------------

# Compute N Largest Drawdowns
def get_max_n_drawdowns(returns, N = 10, log_rets = False, plot = True):
    """ Generate series of drawdowns and plot max N drawdowns.

    Args:
        returns (_type_): _description_
        N (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """

    # If log returns are passed, compute cumulative returns with addition
    if log_rets:
        cum_returns = returns.cumsum()
        cum_returns = (1+cum_returns)
    # If raw returns are passed, compute cumulative returns with multiplication
    else:
        cum_returns = (1 + returns).cumprod()
    
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
    if plot:
        get_n_max_drawdowns.plot.bar()
    
    return get_n_max_drawdowns


# Compute Time-Series of Drawdowns
def get_drawdowns(returns, log_rets = False):
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

    # If log returns are passed, compute cumulative returns with addition
    if log_rets:
        cum_returns = returns.cumsum()
        cum_returns = (1+cum_returns)
    # If raw returns are passed, compute cumulative returns with multiplication
    else:
        cum_returns = (1 + returns).cumprod()

    # Loop through each day's cumulative return
    for date, cum_ret in cum_returns.items():
        
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
def VaR(strategy_returns: pd.Series, alpha = .01, use_laplace = True):
    """ Compute daily Value at Risk & Conditional Value at Risk from strategy/portfolio returns.

        Formula: VaR = Fx^-1 (alpha, loc = mu, scale = sigma) or Inverse Norm CDF (alpha)  --> using inverse of normal cdf gives a closed form solution for a "alpha" probability.
        Formula: CVaR = VaR mu - sigma * phi_pdf(VaR) / phi_cdf(VaR) --> using Mill's Ratio as a closed form solution for a "alpha" probability.
        
        See more at: https://www.youtube.com/watch?v=JUocSFe-DT0
        For further implementations, utilize alt distributions: https://www.youtube.com/watch?v=icC5Z5FM_Sw, https://www.youtube.com/watch?v=zR_liKniGOc
        These could include: laplace, hypersecant, cauchy, logistic, t 

    Args:
        strategy_returns (pd.Series): time series of daily strategy returns.
        alpha (int, optional): _description_. Defaults to 5.
        use_laplace (bool, optional): determines if VaR is computed using Normal or Laplace distribution.

        

    Returns:
        (float, float): VaR, CVaR
    """

    # Leverage parametric quantile function to get the inverse cdf of the standard normal distribution at prob (alpha) --> Z-Score
    z_score = norm.ppf(alpha)
    mu = strategy_returns.mean()
    sigma = strategy_returns.std()
    
    # If we are estimating VaR based on more kurtotic distributions, use Laplace:
    if use_laplace:
        
        # Simulate Laplace Distribution
        simulated_returns = np.random.laplace(loc = mu, scale = sigma, size = 10000)
        VaR = np.percentile(simulated_returns, alpha*100)
        CVaR = simulated_returns[simulated_returns < VaR].mean()

        return VaR, CVaR
    
    # VaR = worst daily return given a particular probability (alpha) given a distribution of returns
    VaR = z_score * sigma + mu  # Alternatively, norm.ppf(alpha, loc = mu, scale = sigma) to avoid z-score normalization and generate it more quickly
    
    # CVaR = expected shortfall below a particular probability threshold (alpha).
    CVaR = mu - sigma * (norm.pdf(z_score) / norm.cdf(z_score))

    return VaR, CVaR

# VaR & CVaR from Bootstrap Simulations
def bootstrap_VaR(bootstrap_returns: pd.DataFrame, alpha=99):
    """Compte Value at Risk & Conditional Value at Risk from series of synthetic/bootstrapped data.
    Args:
        bootstrap_returns (pd.DataFrame of np.array like): multiple synthetic time series.
        alpha (float, optional): confidence level for VaR. Defaults to 0.99.

    Returns:
        (VaR, CVaR): Value at Risk, Conditional Value at Risk.
    """
    
    # Clean returns
    bootstrap_returns = bootstrap_returns.dropna()

    # Value at Risk
    VaR = np.percentile(bootstrap_returns, q=100-alpha)
    
    # Conditional Value at Risk (Expected Shortfall)
    CVaR = np.mean(np.mean(bootstrap_returns[bootstrap_returns<VaR]))

    return VaR, CVaR

# Vol Limit & Conditional Vol Limit (EV of Vol)
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
    trailing_vol = (scaled_returns.rolling(lookback_period).std() * 252 **.5).dropna()

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

# New Custom Bootstrap Method (BEST)
def asymmetric_bootstrap(returns: pd.Series, n_samples=10000):
    """ STILL IN RESEARCH PROCESS: default to Wild Bootstrap when executing strategy robustness tests.
    
        Generate synthetic time series by adding asymmetric (skewed) noise to original time series. This differs from
        the original implementation of Wild Bootstrap in that instead of using (1, -1) or a standard distribution to
        generate noise, it is produced by rescaling returns according to their skew, normalizing these returns which 
        enables the noise to have a mean = 0, stdev = 1, and be rescaled according to randomly sampled returns which mimicks 
        the original implementation's noise component.

        In the original construction of Wild Bootstraps, skew was not sufficiently captured. With the Asymmetric Wild Bootstrap, 
        skew is directly considered in the computation of noise, more efficiently capture the dynamics of financial time series 
        and posterior distribution.

    Args:
        returns (pd.Series): time series of returns.
        n_samples (int, optional): number of synthetic time series generated. Defaults to 10000.
    """

    returns.dropna(inplace=True)
    
    # Compute size of returns series
    n = len(returns)
           
    # Initialize bootstrap samples matrix
    asymmetric_wild_bootsrapped_samples = np.empty((n_samples, n))

    # Nonparametric skew
    # skew = (returns.mean() - returns.median())/returns.std() 
    skew = stats.skew(returns) / 2

    # Generate n_samples of length n
    for i in range(0, n_samples):

        # Scale returns series according to its skew
        if skew > 0: 
            neg_samples = np.abs(returns)*-1
            pos_samples = np.abs(returns)*(1+skew)
        else:
            neg_samples = np.abs(returns)*-1*(1-skew)
            pos_samples = np.abs(returns)*1

        asymmetric_returns = np.array([neg_samples, pos_samples]).reshape(neg_samples.shape[0]*2) 

        # Bootstrap asymmetrically scaled returns and normalize via Z-Score
        wild_noise = np.random.choice(asymmetric_returns, size=n, replace=True)
        wild_noise = (wild_noise - wild_noise.mean()) / wild_noise.std()

        # Scale noise with absolute value of newly bootstrapped skewed returns
        wild_noise = np.abs(np.random.choice(asymmetric_returns, size=n, replace=True)) * wild_noise

        # Add noise to original returns
        asymmetric_wild_bootsrapped_samples[i] = (returns + wild_noise).values

    return pd.DataFrame(asymmetric_wild_bootsrapped_samples, columns=returns.index).T
    

# Bootstrap Method for Heteroskedasticity 
def wild_bootstrap_original(returns: pd.Series, n_samples=10000, normal=True):
    """
    Perform Wild Bootstrap on a 1D array of financial time series data. 
    
    The Wild Bootstrap method is a modification of the classic Bootstrap method that is used to address the issue of 
    heteroscedasticity (i.e., non-constant variance) and volatility clustering in financial time series data. 
    
    The wild bootstrap method involves generating new samples by resampling from the original data and
    multiplying each observation by a random noise value drawn from a standard probability distribution (e.g., Gaussian, Student’s T, Laplace). 
    
    This distribution is chosen to reflect the estimated variance of the original data. The Wild Bootstrap method can improve the 
    accuracy of statistical moment estimation and confidence intervals when dealing with heteroscedastic financial time series.

    Args:
    returns (numpy.ndarray or pd.Series): A 1D array of data.
    n_samples (int): The number of resamples to generate.
    normal (bool): Determines the probability distribution of the noise element (Gaussian or Laplace).

    For more information on Laplace Distribution: https://www.statisticshowto.com/laplace-distribution-double-exponential/
    For more information on Wild Bootstrap Method: https://stats.stackexchange.com/questions/408651/intuitively-how-does-the-wild-bootstrap-work
    
    Returns:
    pd.DataFrame: An matrix of bootstrapped returns.
    """

    returns.dropna(inplace=True)

    # Compute size of returns series
    n = len(returns)

    # Initialize bootstrap samples matrix
    wild_bootsrapped_samples = np.empty((n_samples, n))    

    # Generate n_samples of length n
    for i in range(0, n_samples):
        
        # ------------------------------------------- Generate Noise -------------------------------------------
        if normal:
            # Generate random perturbations from a standardized distribution (e.g., Standard Normal)
            perturbations = np.random.standard_normal(n)

        else: 
            laplace_mu = 0
            laplace_beta =  np.sqrt(np.var(returns)/2)
            # perturbations = np.random.laplace(loc=laplace_mu, scale=laplace_beta, size=n)
            perturbations = np.random.laplace(loc=laplace_mu, scale=1, size=n)

        # ------------------------------------------- Scale Noise -------------------------------------------
        # Randomly sample returns (shuffle) from empirical distribution and scale by Gaussian perturbations
        
        # Scale noise element by bootstrapped returns
        wild_noise = np.random.choice(returns, size=n, replace=True) * perturbations
        
        # ------------------------------------------- Generate Samples -------------------------------------------
        # Add the wild noise to the original data
        wild_bootsrapped_samples[i] = (returns + wild_noise).values     
        
    return pd.DataFrame(wild_bootsrapped_samples, columns=returns.index).T

# Bootstrap Summarizer
def bootstrap_summary(synthetic_data: pd.DataFrame, empirical_data: pd.Series):
    """ Function to directly compare a matrix of bootstrapped time series and its underlying historical/empirical data.
        Relevant features include computing statistical moments of estimations, statistical significance statistics, etc.
        
    Args:
        synthetic_data (pd.DataFrame): _description_
        empirical_data (pd.Series): _description_

    Returns:
        _type_: _description_
    """
    # Change desired formatting
    pd.options.display.float_format = '{:.4f}'.format

    # Declare temp hash tables to store relevant statistics
    bootstrap_summary_data = {}
    empirical_summary_data = {}
            
    # Mean
    mean = np.mean(np.mean(synthetic_data))
    bootstrap_summary_data['Mean'] = mean
    empirical_summary_data['Mean'] = np.mean(empirical_data)

    # Standard Deviation
    mean_std = np.mean(np.std(synthetic_data))
    bootstrap_summary_data['STD'] = mean_std
    empirical_summary_data['STD'] = np.std(empirical_data)

    # Mean Skew
    mean_skew = np.mean(stats.skew(synthetic_data))
    bootstrap_summary_data['Skew'] = mean_skew
    empirical_summary_data['Skew'] = stats.skew(empirical_data)

    # Mean Kurtosis
    mean_kurtosis = np.mean(stats.kurtosis(synthetic_data, fisher=False))
    bootstrap_summary_data['Kurtosis'] = mean_kurtosis
    empirical_summary_data['Kurtosis'] = stats.kurtosis(empirical_data, fisher=False)

    # Mean 5'th Moment (Asymmetry of Tails)
    mean_5th_moment = np.mean(stats.moment(synthetic_data, moment=5))
    bootstrap_summary_data['5th Moment'] = mean_5th_moment
    empirical_summary_data['5th Moment'] = stats.moment(empirical_data, moment=5)

    # Value at Risk
    bootstrap_summary_data['VaR - 99% CI'], bootstrap_summary_data['CVaR - 99% CI'] = bootstrap_VaR(synthetic_data)

    # P-Value
    synth_cum_rets = (np.exp(np.log(synthetic_data+1).cumsum().iloc[:, 0:1000])-1).iloc[-1,:] # use log returns for faster computations
    empirical_cum_rets = ((1+empirical_data).cumprod()-1).iloc[-1]
    p_value = len(synth_cum_rets[synth_cum_rets>empirical_cum_rets]) / len(synth_cum_rets)
    empirical_summary_data['P-Value'] = p_value

    # Expected Value of Cumulative Returns
    bootstrap_summary_data['EV of Cumulative Returns'] = synth_cum_rets.mean()

    # % of Negative of Cumulative Returns
    bootstrap_summary_data['Pct. of Negative Cumulative Returns'] = len(synth_cum_rets.where(synth_cum_rets<0).dropna()) / len(synth_cum_rets)

    # Standard Deviation of Mean
    std_mean = np.std(np.mean(synthetic_data))
    bootstrap_summary_data['STD of Mean'] = std_mean

    # Standard Deviation of Median
    std_median = np.std((synthetic_data).median())
    bootstrap_summary_data['STD of Median'] = std_median

    # Mean Squared Error
    mse = ((synthetic_data.mean() - empirical_data.mean())**2).mean()
    bootstrap_summary_data['MSE'] = mse

    # R^2 = Average R^2 Across All Samples
    r_2 = []

    for i, data in synthetic_data.items():
        tmp_r_2 = 1 - np.sum((data - empirical_data)**2) / np.sum((empirical_data - empirical_data.mean())**2)
        r_2.append(tmp_r_2)

    bootstrap_summary_data['R^2'] = np.mean(r_2)

    # Summarize Data
    bootstrap_summary_data = pd.Series(bootstrap_summary_data, name='Bootstrap Summary Statistics')
    empirical_summary_data = pd.Series(empirical_summary_data, name='Historical Summary Statistics')

    summary_data = pd.concat([bootstrap_summary_data, empirical_summary_data], axis=1)
    
    return summary_data


def robustness_test(returns: pd.Series):
    """ Function to conduct stress-tests on a backtest's time series. This includes:

        1) Bootstrap Monte Carlo Simulation of Returns (Leveraging Asymmetric/Wild Bootstrap which better 
           capture the dynamics of financial time series).

        2) Comparative Analysis on MC vs. Empirical Data.

        3) Summary Statistics: P-Value, Mean, STD, Skew, Kurtosis, 5th Moment, 
           STD of MC Mean, STD of MC Median, MSE, R^2, VaR, EV of Cum Rets, & % Negative Cumulative Returns.

        4) Plot of Cumulative Simulated Returns.

    Args:
        returns: None
    """

    # Leverage Asymmetric Wild Bootstrap Algorithm
    # asymmetric_bootstrap_returns = asymmetric_bootstrap(returns, n_samples=10000)
    # print(bootstrap_summary(synthetic_data=asymmetric_bootstrap, empirical_data=returns))

    # # Compute MC Simulation Cumulative Returns
    # asymmetric_bootstrap_cum_rets = (np.exp(1+cumulative_returns(np.log(1+asymmetric_bootstrap_returns.iloc[:, 0:1000]), log_rets=True))-1)
    # asymmetric_bootstrap_cum_rets.plot(title=f'{returns.name} - Asymmetric Boostrap Returns', legend=False)
    # plt.show()

    # Leverage Original Wild Bootstrap Algorithm
    wild_bootstrap_returns = wild_bootstrap_original(returns, n_samples=10000)

    # Compute MC Simulation Cumulative Returns
    wild_bootstrap_cum_rets = (np.exp(1+pt.cumulative_returns(np.log(1+wild_bootstrap_returns.iloc[:, 0:1000]), log_rets=True))-1)
    wild_bootstrap_cum_rets.plot(title=f'{returns.name} - Wild Boostrap Monte Carlo Returns', legend=False)
    plt.show()

    return bootstrap_summary(synthetic_data=wild_bootstrap_returns, empirical_data=returns)


# ------------------------------------------------------------------------- Dead Code -------------------------------------------------------------------------

# def value_at_risk(returns, percentile = 5):
#     """ Compute Value at Risk & Conditional Value at Risk from strategy/portfolio returns

#     Args:
#         returns (pd.DataFrame): DataFrame of returns from a Monte Carlo simulation.
#         percentile (int): Var & CVar confidence level. Defaults to 5th percentile for 95% confidence.
#     """
#     if type(returns) == pd.DataFrame:
#         # Compute daily Var & CVar at "percentile" confidence interval
#         sim_daily_rets = returns.values.tolist()[0]
#         Var = np.percentile(sim_daily_rets, percentile)
#         # Convert all daily returns to np.array
#         tmp_CVar_array = np.array(sim_daily_rets)
#         # Isolate returns < Var
#         tmp_CVar_array = np.where(tmp_CVar_array < Var, tmp_CVar_array, np.nan)
#         # Delete NaNs and compute mean ~ CVar
#         CVar = tmp_CVar_array[~np.isnan(tmp_CVar_array)].mean()

#     elif type(returns) == pd.Series:
#         # Compute daily Var & CVar at "percentile" confidence interval
#         sim_daily_rets = returns.values.tolist()
#         Var = np.percentile(sim_daily_rets, percentile)
#         # Convert all daily returns to np.array
#         tmp_CVar_array = np.array(sim_daily_rets)
#         # Isolate returns < Var
#         tmp_CVar_array = np.where(tmp_CVar_array < Var, tmp_CVar_array, np.nan)
#         # Delete NaNs and compute mean ~ CVar
#         CVar = tmp_CVar_array[~np.isnan(tmp_CVar_array)].mean()

#     return Var, CVar


# # Monte Carlo Simulations (Var, CVar)
# def monte_carlo_VaR (returns, days = 10, N = 1000, percentile = 5, use_students_t = False, plot=False):
#     """ Conduct Monte Carlo simulations on a given time-series of strategy returns data. This will 
#         enable the researcher to acquire relevant risk metrics and provide insight into the distribution of returns. 

#     Args:
#         returns (pd.Series): Series of returns data.
#         days (int, optional): Duration of Monte-Carlo simulation to be plotted. Defaults to 10.
#         N (int, optional): Number of simulations. Defaults to 1000.
#         percentile (int, optional): Var & CVar condifence level. Defaults to 5th percentile for 95% confidence.
#         use_students_t (bool, optional): Determines if the simulation will utilize both normal and student's t distributions. Defaults to False.
#         plot (bool, optional): Determines if simulations will be plotted. Defaults to False.
        

#     Returns:
#         _type_: _description_
#     """

#     # Plot Empirical Returns Distribution
#     plt.hist(returns, 50)
    
#     # Get daily returns' mean, std, kurtosis, skew, & df
#     mu = returns.mean()
#     sigma = returns.std()
#     strat_kurtosis = kurtosis(returns, fisher=False)
#     strat_skew = skew(returns)
#     # Compute Student's T Distribution's "Degrees of Freedom" = 6/kurt + 4
#     df = 6 / strat_kurtosis + 4

#     # Nomral Distribution Monte Carlo Simulation on N iterations
#     mc_norm = pd.DataFrame(np.random.normal(mu, sigma, (days, N)))
#     # Compute Var & CVar
#     norm_Var, norm_CVar = value_at_risk(mc_norm, percentile=percentile)
#     full_sample_Var, full_sample_CVar = value_at_risk(mc_norm.cumsum().iloc[-1], percentile)
    
#     if use_students_t == True:
#         mc_t_standard = pd.DataFrame(np.array(t.rvs(df, size=(days, N)))) #1 Compute Student's T MC
#         t_Var, t_CVar = value_at_risk(mc_t_standard, percentile=percentile)   
#         print(f'T-Var: {t_Var}... T-CVar: {t_CVar}')
#         # Undo Z-Score: 
#         # mc_t = mc_t_standard/np.sqrt(df/(df-2)) # 2
#         # mc_t = mc_t * sigma # 3
#         # mc_t = mc_t + mu # 4
#         # t_plot = plt.hist(np.sort(mc_t), 50)
        
#     # Plot Normal Distribtuion MC result
#     # norm_plot = plt.hist(mc_norm, 50)
#     # plt.show()
#     if plot == True:
#         mc_norm_cumsum = mc_norm.cumsum().plot(title='Normal Distribution MC Sim', legend=False)
#         plt.show()

#         # Plot Student's T MC Results
#         if use_students_t == True:
#             mc_t_standard_cumsum = mc_t_standard.cumsum().head(252).plot(title='Student T Distribution MC Sim', legend=False)
#             plt.show()


#     return norm_Var, norm_CVar, full_sample_Var, full_sample_CVar # mu, sigma, strat_kurtosis, df, mc_norm,
    
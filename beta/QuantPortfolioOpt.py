'''
Library for walk-forward portfolio optimization.

Current Implementations:
1) Generalized Class for Portfolio Optimization
2) Portfolio Stress Testing Methods (Rebalancing Timing Luck)

'''


from quant_tools import risk_analysis as ra, performance_analysis as pt, data_preprocessing as dp
from quant_tools.beta import opt_functions as opt, cov_functions as risk_models
from scipy.optimize import Bounds
import statsmodels.api as sm
from scipy import stats
import pandas as pd
import numpy as np


class walk_forward_portfolio_optimization():
    def __init__(self, hist_returns: pd.DataFrame, args: tuple, rebal_freq: int = 21, optimization_method: str = "Max Sharpe Ratio", cov_method: str = "ewma_cov"):
        
        self.hist_returns = hist_returns     
        self.expected_returns = self.hist_returns.rolling(252).mean() # self.get_expected_returns(self.hist_returns) -- perhaps user passes these into function
        self.rebal_freq = rebal_freq
        self.optimization_method = optimization_method
        self.cov_method = cov_method
        # self.constraints -- implement later

        self.optimization_algo_map = {  "Unconstrained Max Sharpe Ratio" : opt.unconstrained_max_sharpe_mvo,
                                        "Max Sharpe Ratio" : opt.max_sharpe_mvo,
                                        "Risk Parity" : opt.risk_parity, 
                                        "Dollar Risk Parity" : opt.dollar_risk_parity,
                                        "ATR Risk Parity" : opt.atr_risk_parity,
                                     }
        
        self.optimization_algo = self.optimization_algo_map[self.optimization_method]

        self.cov_algo = risk_models.risk_matrix[cov_method]

        if args:
            self.args = args
        else:
            self.args = {   "long_only": False,
                            "vol_target": 0.01,
                            "max_position_weight": 0.2,
                            "net_exposure": 1,
                            "market_bias": 0,
                            "constrained": True,
                            "verbose": False
                        }
            
        self.w = self.run()
    
    def clean_args(self, args: dict) -> dict:
                
        # Get opt function args
        supported_args = list(self.optimization_algo.__code__.co_varnames[:self.optimization_algo.__code__.co_argcount])

        # Drop unsupported args
        unsupported_args = [key for key in args.keys() if key not in supported_args]
        if len(unsupported_args) > 0: 
            print(f"{unsupported_args} are not supported args in the {self.optimization_method} optimization function!")
            print(f"Supported args are {supported_args}.")
            for key in unsupported_args:
                del args[key]

        # Get required opt function args (slice args for required args)
        required_args = list(self.optimization_algo.__code__.co_varnames[:self.optimization_algo.__code__.co_argcount])

        # Check if all required args are defined
        missing_required_args = [key for key in required_args if key not in args.keys()]
        if len(missing_required_args) > 0:
            raise TypeError(f"{self.optimization_method} optimization function missing required argument(s):\n{missing_required_args}")
        
        return args
    
    def run(self):
        
        # Empty w matrix for indexing purposes
        empty_w = pd.DataFrame(index=self.hist_returns.index)

        # Hash Map to hold walk-forward optimized weights 
        w = {}

        if self.rebal_freq is not None:
            for date in self.hist_returns.index[::self.rebal_freq]:               
                                
                # Get expanding hist & expected returns
                tmp_hist_returns = self.hist_returns.loc[:date].dropna()
                tmp_expected_returns = self.expected_returns.loc[date].dropna()
                # tmp_cov = self.cov_algo(hist_returns=tmp_hist_returns) -- implement functionality later
                
                # Update & clean args
                self.args.update({"hist_returns" : tmp_hist_returns, "expected_returns" : tmp_expected_returns})
                # self.args.update({"hist_returns" : tmp_hist_returns, "expected_returns" : tmp_expected_returns, "cov" : tmp_cov}) -- implement functionality later
                self.clean_args(self.args)                
                
                # Get optimal weights
                w[date] = self.optimization_algo(**self.args)
            
            # Fill weights
            w = pd.concat([pd.DataFrame(w).T, empty_w], axis=1).ffill()
            
        return w
    
    def get_cov(self, hist_returns: pd.DataFrame, cov_method: str):
        # implement library later
        cov = hist_returns.cov()
        return cov
    
    def get_expected_returns(self):
        # implement library later
        return None

    


        

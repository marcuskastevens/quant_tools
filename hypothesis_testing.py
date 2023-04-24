import pandas as pd
import numpy as np
import sklearn
import scipy.stats as stats
import statsmodels.api as sm

def two_sample_t_test(mu_1, s_1, n_1, mu_2, s_2, n_2):

    # Calculate the test statistic and p-value
    t_stat, p_value = stats.ttest_ind_from_stats(mean1=mu_1, std1=s_1, nobs1=n_1, mean2=mu_2, std2=s_2, nobs2=n_2)

    # Print the results
    print(f"t-stat: {t_stat:.2f}")
    print(f"p-value: {p_value:.4f}")

    # Compare p-value with significance level (e.g., 0.01)
    if p_value < 0.01:
        print("Reject null hypothesis: There is a statistically significant difference.")
    else:
        print("Fail to reject null hypothesis: There is no statistically significant difference.")


def regression_beta_two_tailed_ttest(beta_1, se_1, beta_2, se_2, min_nobs):
    """ Conduct two-tailed t-test on two linear regression models to determine if there is a 
        statistically significant difference between their betas.


    Args:
        beta_1 (_type_): _description_
        se_1 (_type_): _description_
        beta_2 (_type_): _description_
        se_2 (_type_): _description_
        min_nobs (_type_): _description_
    """

    # Calculate the t-statistic and p-value
    se_diff = np.sqrt(se_1**2 + se_2**2)
    t_stat = (beta_1 - beta_2) / se_diff
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=min_nobs-2))

    # Print the results
    print(f"t-stat: {t_stat:.2f}")
    print(f"p-value: {p_value:.4f}")
    # Compare p-value with significance level (e.g., 0.01)
    if p_value < 0.01:
        print("Reject null hypothesis: There is a statistically significant difference.")
    else:
        print("Fail to reject null hypothesis: There is no statistically significant difference.")

def regression_beta_one_tailed_ttest(beta_1, se_1, nobs_1, beta_2, se_2, nobs_2):
    """ Conduct one-tailed t-test on two linear regression models to determine if beta_1 is statistically
        greater than beta_2.

    Args:
        beta_1 (_type_): _description_
        se_1 (_type_): _description_
        beta_2 (_type_): _description_
        se_2 (_type_): _description_
        min_nobs (_type_): _description_
    """

    # Calculate the t-statistic and p-value
    se_diff = np.sqrt(se_1**2 + se_2**2)
    t_stat = (beta_1 - beta_2) / se_diff
    p_value = (1 - stats.t.cdf(abs(t_stat), df=(nobs_1+nobs_2)-2)

    # Print the results
    print(f"t-stat: {t_stat:.2f}")
    print(f"p-value: {p_value:.4f}")
    # Compare p-value with significance level (e.g., 0.01)
    if p_value < 0.01:
        print("Reject null hypothesis: There is a statistically significant difference.")
    else:
        print("Fail to reject null hypothesis: There is no statistically significant difference.")
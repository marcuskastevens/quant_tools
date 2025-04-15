"""
Module implementing bet sizing utility functions.
"""


def calculate_second_order_kelly_leverage(mu: float, sigma: float) -> float:
    """
    Calculates leverage ratio that asymptotically maximizes geometric growth rate (log wealth) via a second order Taylor approximation.
    See Chapter 9 of "Positional Options Trading" by Euan Sinclar for a derivation.
    """
    return mu / sigma**2


def calculate_third_order_kelly_leverage(mu: float, sigma: float, skew: float) -> float:
    """
    Calcualtes leverage ratio that asymptotically maximizes geometric growth rate (log wealth) via a third order Taylor approximation (i.e., accounts for skew).
    See Chapter 9 of "Positional Options Trading" by Euan Sinclar for a derivation.
    """
    return calculate_second_order_kelly_leverage(mu=mu, sigma=sigma) + skew * sigma**3 * mu**2 / (mu**2 + sigma**2) ** 3

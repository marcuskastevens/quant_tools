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


def calculate_poker_kelly_bet_size(equity: float, pot_size: float, bankroll: float) -> float:
    """
    Calculates the heads-up (i.e., 1v1) Kelly-optimal bet size assuming the opponent calls after a bet.

    This is a simplified formula that omits the opponent's fold probability, dynamic post-call equity,
    the opponent's response curves, and various discretionary elements that impact bet sizing.

    This formula is derived by modeling log wealth as a function of equity, bet size, pot size, and bankroll, and optimizing it with respect to bet size.

    Define log growth function:
        g(bankroll) = equity * ln(bakroll + pot_size + bet_size) + (1 - equity) * ln(bankroll - bet_size) - ln(bankroll)

    Differentiate with respect to bet size:
        d_g/d_bet_size = equity / (bakroll + pot_size + bet_size) - (1 - equity) / (bankroll - bet_size)

    Set d_g/d_bet_size = 0 and solve for the optimal bet size:
        equity / (bakroll + pot_size + bet_size) =  (1 - equity) / (bankroll - bet_size)
        equity * (bankroll - bet_size) =  (1 - equity) * (bakroll + pot_size + bet_size)
        equity * bankroll - equity * bet_size =  (1 - equity) * bakroll + (1 - equity) * pot_size + (1 - equity) * bet_size
        equity * bankroll - (1 - equity) * bakroll - (1 - equity) * pot_size = equity * bet_size + (1 - equity) * bet_size
        equity * bankroll - (1 - equity) * bakroll - (1 - equity) * pot_size = bet_size * (equity + (1 - equity))
        --> bet_size = bankroll * (2 * equity - 1) - (1 - equity) * pot_size
        --> % bet_size = (2 * equity - 1) - (1 - equity) * pot_size / bankroll

    Notice, when the pot size is very large, our optimal bet size shrinks. This is due to the diminishing marginal utility of betting.
    Strong equity in large pot size situations implies already good pot odds, thus, the marginal benefit we get from risking more is diminished.
    """
    return (2 * equity - 1) - (1 - equity) * pot_size / bankroll

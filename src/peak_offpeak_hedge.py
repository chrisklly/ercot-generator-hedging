import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def make_peak_mask(datetimes):
    x = pd.DataFrame({"datetime": datetimes})
    hour = x["datetime"].dt.hour
    weekday = x["datetime"].dt.weekday

    is_peak = (
        (weekday < 5) &
        (hour >= 6) &
        (hour <= 21)
    )

    return is_peak.values


def asian_put_by_block(price_paths, mask, strike, annual_rate=0.04):
    """
    price_paths: n_paths x n_hours
    mask: Boolean array selecting peak or off-peak hours
    """
    block_prices = price_paths[:, mask]
    avg_block_price = block_prices.mean(axis=1)

    payoff_per_mwh = np.maximum(strike - avg_block_price, 0)

    n_hours = price_paths.shape[1]
    tau = n_hours / (365 * 24)
    discount = np.exp(-annual_rate * tau)

    premium_per_mwh = discount * payoff_per_mwh.mean()

    return {
        "avg_block_price": avg_block_price,
        "payoff_per_mwh": payoff_per_mwh,
        "premium_per_mwh": premium_per_mwh,
    }


def optimize_two_block_hedge(
    unhedged_profit,
    peak_payoff,
    offpeak_payoff,
    peak_premium,
    offpeak_premium,
    peak_notional,
    offpeak_notional,
):
    """
    Optimize peak and off-peak hedge ratios jointly.
    """

    def objective(x):
        h_peak, h_offpeak = x

        hedged = (
            unhedged_profit
            + h_peak * peak_notional * peak_payoff
            + h_offpeak * offpeak_notional * offpeak_payoff
            - h_peak * peak_notional * peak_premium
            - h_offpeak * offpeak_notional * offpeak_premium
        )

        var5 = np.percentile(hedged, 5)
        es5 = hedged[hedged <= var5].mean()

        return -es5

    # Simple grid search is more transparent for a course project
    grid = np.linspace(0, 1.5, 31)

    best = None

    for hp in grid:
        for ho in grid:
            val = objective((hp, ho))
            if best is None or val < best["objective"]:
                best = {
                    "peak_hedge_ratio": hp,
                    "offpeak_hedge_ratio": ho,
                    "objective": val,
                }

    return best


def block_generation_notional(dispatch_mw, mask):
    """
    dispatch_mw: n_paths x n_hours
    returns expected MWh generated in peak/off-peak block.
    """
    return dispatch_mw[:, mask].sum(axis=1).mean()
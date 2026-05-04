import numpy as np
import pandas as pd

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["datetime"].dt.date
    out["month"] = out["datetime"].dt.to_period("M").astype(str)
    out["hour"] = out["datetime"].dt.hour
    out["dayofyear"] = out["datetime"].dt.dayofyear
    return out

def estimate_mean_reverting_jump_model(price_series: pd.Series, clip_extremes=False):
    """
    Estimate a simple discrete mean-reverting model:
        dP_t = a + b P_t + e_t

    Then identify jump residuals as residuals larger than 3 sigma.

    This is intentionally simple and explainable for a course project.
    """
    p = price_series.dropna().astype(float).values
    if len(p) < 100:
        raise ValueError("Need at least 100 observations for parameter estimation.")

    x = p[:-1]
    y = np.diff(p)

    if clip_extremes:
        lo, hi = np.percentile(y, [1, 99])
        y_fit = np.clip(y, lo, hi)
    else:
        y_fit = y

    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y_fit, rcond=None)[0]
    a, b = beta

    kappa = max(-b, 1e-8)
    theta = a / kappa
    residuals = y - X @ beta
    sigma = residuals.std(ddof=1)

    threshold = 3 * sigma
    jumps = residuals[np.abs(residuals) > threshold]

    return {
        "a": float(a),
        "b": float(b),
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "jump_prob": float(len(jumps) / len(residuals)),
        "jump_mean": float(jumps.mean()) if len(jumps) else 0.0,
        "jump_std": float(jumps.std(ddof=1)) if len(jumps) > 1 else 0.0,
        "n_jumps": int(len(jumps)),
        "residual_std": float(sigma),
    }

def simulate_hourly_prices(
    p0,
    params,
    n_hours,
    n_paths,
    seed=123,
    floor=-250.0,
    cap=5000.0,
):
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_paths, n_hours + 1))
    paths[:, 0] = p0

    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    jump_prob = params.get("jump_prob", 0.0)
    jump_mean = params.get("jump_mean", 0.0)
    jump_std = params.get("jump_std", 0.0)

    for t in range(1, n_hours + 1):
        eps = rng.normal(0, sigma, n_paths)
        jump_flag = rng.binomial(1, jump_prob, n_paths)
        if jump_std > 0:
            jump_size = rng.normal(jump_mean, jump_std, n_paths)
        else:
            jump_size = jump_mean
        paths[:, t] = paths[:, t-1] + kappa * (theta - paths[:, t-1]) + eps + jump_flag * jump_size
        paths[:, t] = np.clip(paths[:, t], floor, cap)

    return paths[:, 1:]

def simulate_gas_prices(
    gas0=3.0,
    n_hours=720,
    n_paths=10000,
    annual_vol=0.45,
    mean_reversion=0.02,
    long_run=3.25,
    seed=456,
):
    """
    Simple hourly gas price simulator in $/MMBtu.
    For first project version, this can be replaced with a constant gas price.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / (365 * 24)
    sigma_h = annual_vol * np.sqrt(dt)
    gas = np.zeros((n_paths, n_hours))
    gas[:, 0] = gas0
    for t in range(1, n_hours):
        shock = rng.normal(0, sigma_h, n_paths)
        gas[:, t] = gas[:, t-1] + mean_reversion * (long_run - gas[:, t-1]) * dt + gas[:, t-1] * shock
        gas[:, t] = np.clip(gas[:, t], 0.25, 50)
    return gas

def marginal_cost(gas_price, heat_rate=7.0, variable_om=3.0):
    return heat_rate * gas_price + variable_om

def dispatch_mw(
    price,
    cost,
    capacity_mw,
    availability,
    width=5.0,
    capacity_factor_cap=0.90,
):
    """
    Smooth economic dispatch.
    Output approaches zero when price is far below cost and approaches
    capacity * availability * capacity_factor_cap when price is far above cost.
    """
    econ = sigmoid((price - cost) / width)
    return capacity_mw * availability * capacity_factor_cap * econ

def simulate_generator_profit(
    price_paths,
    gas_paths,
    capacity_mw,
    heat_rate=7.0,
    variable_om=3.0,
    availability_mean=0.94,
    availability_std=0.04,
    min_availability=0.70,
    max_availability=1.00,
    dispatch_width=5.0,
    capacity_factor_cap=0.90,
    seed=789,
):
    rng = np.random.default_rng(seed)
    n_paths, n_hours = price_paths.shape
    availability = rng.normal(availability_mean, availability_std, (n_paths, n_hours))
    availability = np.clip(availability, min_availability, max_availability)

    cost = marginal_cost(gas_paths, heat_rate, variable_om)
    q = dispatch_mw(
        price_paths, cost, capacity_mw, availability,
        width=dispatch_width, capacity_factor_cap=capacity_factor_cap
    )

    hourly_profit = q * (price_paths - cost)
    monthly_profit = hourly_profit.sum(axis=1)
    monthly_generation_mwh = q.sum(axis=1)

    return {
        "hourly_profit": hourly_profit,
        "monthly_profit": monthly_profit,
        "monthly_generation_mwh": monthly_generation_mwh,
        "dispatch_mw": q,
        "marginal_cost": cost,
    }

def asian_put_payoff(price_paths, strike):
    avg_price = price_paths.mean(axis=1)
    return np.maximum(strike - avg_price, 0), avg_price

def price_asian_put(price_paths, strike, annual_rate=0.04, hours=720):
    payoff, avg_price = asian_put_payoff(price_paths, strike)
    tau = hours / (365 * 24)
    discount = np.exp(-annual_rate * tau)
    price = discount * payoff.mean()
    stderr = discount * payoff.std(ddof=1) / np.sqrt(len(payoff))
    return {
        "premium_per_mwh": price,
        "stderr_per_mwh": stderr,
        "payoff_per_mwh": payoff,
        "avg_price": avg_price,
    }

def risk_metrics(x):
    x = np.asarray(x)
    var5 = np.percentile(x, 5)
    es5 = x[x <= var5].mean()
    return {
        "mean": x.mean(),
        "std": x.std(ddof=1),
        "var_5": var5,
        "expected_shortfall_5": es5,
        "probability_loss": (x < 0).mean(),
        "p01": np.percentile(x, 1),
        "p50": np.percentile(x, 50),
        "p95": np.percentile(x, 95),
    }

def hedge_analysis(unhedged_profit, price_paths, generation_mwh, strike, hedge_ratio, annual_rate=0.04):
    option = price_asian_put(price_paths, strike, annual_rate=annual_rate, hours=price_paths.shape[1])
    payoff_per_mwh = option["payoff_per_mwh"]
    premium_per_mwh = option["premium_per_mwh"]

    hedge_notional = hedge_ratio * np.mean(generation_mwh)
    premium_paid = hedge_notional * premium_per_mwh
    hedge_payoff = hedge_notional * payoff_per_mwh

    hedged_profit = unhedged_profit + hedge_payoff - premium_paid

    u = risk_metrics(unhedged_profit)
    h = risk_metrics(hedged_profit)

    return {
        "strike": strike,
        "hedge_ratio": hedge_ratio,
        "hedge_notional_mwh": hedge_notional,
        "premium_per_mwh": premium_per_mwh,
        "premium_paid": premium_paid,
        "mean_unhedged": u["mean"],
        "mean_hedged": h["mean"],
        "std_unhedged": u["std"],
        "std_hedged": h["std"],
        "var5_unhedged": u["var_5"],
        "var5_hedged": h["var_5"],
        "es5_unhedged": u["expected_shortfall_5"],
        "es5_hedged": h["expected_shortfall_5"],
        "prob_loss_unhedged": u["probability_loss"],
        "prob_loss_hedged": h["probability_loss"],
        "variance_reduction": 1 - np.var(hedged_profit, ddof=1) / np.var(unhedged_profit, ddof=1),
        "var5_improvement": h["var_5"] - u["var_5"],
        "es5_improvement": h["expected_shortfall_5"] - u["expected_shortfall_5"],
        "hedged_profit": hedged_profit,
    }

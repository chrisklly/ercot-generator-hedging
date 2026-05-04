import numpy as np
import pandas as pd


def fit_seasonal_model(df):
    """
    Fit a deterministic seasonal component for ERCOT prices.

    The model decomposes price into:
        overall mean
        + hour-of-day effect
        + month-of-year effect
        + weekday effect
        + peak/off-peak effect

    The residual from this decomposition is later modeled stochastically.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["month_num"] = df["datetime"].dt.month
    df["weekday"] = df["datetime"].dt.weekday

    df["is_peak"] = (
        (df["weekday"] < 5)
        & (df["hour"] >= 6)
        & (df["hour"] <= 21)
    )

    overall = df["price"].mean()

    hour_effect = df.groupby("hour")["price"].mean() - overall
    month_effect = df.groupby("month_num")["price"].mean() - overall
    weekday_effect = df.groupby("weekday")["price"].mean() - overall
    peak_effect = df.groupby("is_peak")["price"].mean() - overall

    seasonal = (
        overall
        + df["hour"].map(hour_effect)
        + df["month_num"].map(month_effect)
        + df["weekday"].map(weekday_effect)
        + df["is_peak"].map(peak_effect)
    )

    residual = df["price"] - seasonal

    return {
        "overall": overall,
        "hour_effect": hour_effect,
        "month_effect": month_effect,
        "weekday_effect": weekday_effect,
        "peak_effect": peak_effect,
        "residual": residual,
        "calibration_start": df["datetime"].min(),
        "calibration_end": df["datetime"].max(),
    }


def seasonal_component(datetimes, model):
    """
    Reconstruct deterministic seasonal component for future timestamps.
    """
    x = pd.DataFrame({"datetime": pd.to_datetime(datetimes)})
    x["hour"] = x["datetime"].dt.hour
    x["month_num"] = x["datetime"].dt.month
    x["weekday"] = x["datetime"].dt.weekday

    x["is_peak"] = (
        (x["weekday"] < 5)
        & (x["hour"] >= 6)
        & (x["hour"] <= 21)
    )

    component = (
        model["overall"]
        + x["hour"].map(model["hour_effect"]).fillna(0)
        + x["month_num"].map(model["month_effect"]).fillna(0)
        + x["weekday"].map(model["weekday_effect"]).fillna(0)
        + x["is_peak"].map(model["peak_effect"]).fillna(0)
    )

    return component.values


def estimate_residual_jump_model(
    residuals,
    min_kappa=0.15,
    winsor_limits=(1, 99),
    jump_threshold_sigma=3.0,
    max_jump_multiple=2.0,
):
    """
    Estimate a mean-reverting residual model with controlled jumps.

    Discrete model:
        dR_t = a + b R_t + epsilon_t

    Converted into:
        dR_t = kappa(theta - R_t) + epsilon_t + jump_t

    Calibration choices:
    - Winsorize residual shocks for volatility estimation to avoid extreme
      scarcity events dominating the entire simulation.
    - Cap jump mean and jump standard deviation to prevent persistent
      unrealistic high-price regimes.
    - Enforce minimum mean reversion so simulated residuals return toward
      normal conditions.
    """
    r = pd.Series(residuals).dropna().astype(float).values

    if len(r) < 100:
        raise ValueError("Not enough residual observations to estimate price model.")

    x = r[:-1]
    y = np.diff(r)

    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    a, b = beta

    kappa = max(-b, min_kappa)
    theta = a / kappa

    fitted = X @ beta
    eps = y - fitted

    lo, hi = np.percentile(eps, winsor_limits)
    eps_core = np.clip(eps, lo, hi)

    sigma = eps_core.std(ddof=1)

    jump_threshold = jump_threshold_sigma * sigma
    jumps = eps_core[np.abs(eps_core) > jump_threshold]

    jump_prob = len(jumps) / len(eps_core)

    if len(jumps) > 0:
        jump_mean = jumps.mean()
    else:
        jump_mean = 0.0

    if len(jumps) > 1:
        jump_std = jumps.std(ddof=1)
    else:
        jump_std = 0.0

    jump_mean = float(
        np.clip(
            jump_mean,
            -max_jump_multiple * sigma,
            max_jump_multiple * sigma,
        )
    )

    jump_std = float(min(jump_std, max_jump_multiple * sigma))

    return {
        "a": float(a),
        "b": float(b),
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "jump_prob": float(jump_prob),
        "jump_mean": float(jump_mean),
        "jump_std": float(jump_std),
        "winsor_low": float(lo),
        "winsor_high": float(hi),
        "n_jumps": int(len(jumps)),
    }


def simulate_residual_paths(
    r0,
    params,
    n_hours,
    n_paths,
    seed=42,
):
    """
    Simulate mean-reverting residual paths with occasional controlled jumps.
    """
    rng = np.random.default_rng(seed)

    paths = np.zeros((n_paths, n_hours))
    prev = np.full(n_paths, r0, dtype=float)

    kappa = params["kappa"]
    theta = params["theta"]
    sigma = params["sigma"]
    jump_prob = params["jump_prob"]
    jump_mean = params["jump_mean"]
    jump_std = params["jump_std"]

    for t in range(n_hours):
        noise = rng.normal(0, sigma, n_paths)

        jump_flag = rng.binomial(1, jump_prob, n_paths)

        if jump_std > 0:
            jump = rng.normal(jump_mean, jump_std, n_paths)
        else:
            jump = np.full(n_paths, jump_mean)

        current = (
            prev
            + kappa * (theta - prev)
            + noise
            + jump_flag * jump
        )

        paths[:, t] = current
        prev = current

    return paths


def simulate_seasonal_price_paths(
    df,
    start_datetime,
    n_hours,
    n_paths,
    seed=42,
    calibration_years=5,
    price_floor=-250,
    price_cap=5000,
    min_kappa=0.15,
    winsor_limits=(1, 99),
    jump_threshold_sigma=3.0,
    max_jump_multiple=2.0,
):
    """
    Simulate hourly ERCOT price paths using recent-history seasonal calibration.

    Parameters
    ----------
    df : DataFrame
        Must contain datetime and price columns.
    start_datetime : datetime-like
        First simulated timestamp.
    n_hours : int
        Number of hourly periods to simulate.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed.
    calibration_years : int
        Number of recent years used to estimate seasonality and residual dynamics.
    price_floor : float
        Lower bound for simulated prices.
    price_cap : float
        Upper bound for simulated prices.
    min_kappa : float
        Minimum mean reversion speed.
    winsor_limits : tuple
        Percentiles used to winsorize residual shocks.
    jump_threshold_sigma : float
        Jump classification threshold.
    max_jump_multiple : float
        Cap on jump mean/std as a multiple of residual volatility.

    Returns
    -------
    price_paths : ndarray
        Simulated prices, shape (n_paths, n_hours).
    future_datetimes : DatetimeIndex
        Simulated hourly timestamps.
    params : dict
        Estimated stochastic residual parameters.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["datetime", "price"]).sort_values("datetime")

    if df.empty:
        raise ValueError("Input price dataframe is empty after cleaning.")

    cutoff = df["datetime"].max() - pd.DateOffset(years=calibration_years)
    df_cal = df[df["datetime"] >= cutoff].copy()

    if len(df_cal) < 24 * 180:
        raise ValueError(
            "Not enough recent calibration data. "
            "Use a longer dataset or reduce calibration_years."
        )

    seasonal_model = fit_seasonal_model(df_cal)
    residuals = seasonal_model["residual"]

    params = estimate_residual_jump_model(
        residuals=residuals,
        min_kappa=min_kappa,
        winsor_limits=winsor_limits,
        jump_threshold_sigma=jump_threshold_sigma,
        max_jump_multiple=max_jump_multiple,
    )

    future_datetimes = pd.date_range(
        start=pd.to_datetime(start_datetime),
        periods=n_hours,
        freq="h",
    )

    residual_paths = simulate_residual_paths(
        r0=float(residuals.iloc[-1]),
        params=params,
        n_hours=n_hours,
        n_paths=n_paths,
        seed=seed,
    )

    season = seasonal_component(future_datetimes, seasonal_model)

    price_paths = residual_paths + season.reshape(1, -1)
    price_paths = np.clip(price_paths, price_floor, price_cap)

    params["calibration_start"] = str(seasonal_model["calibration_start"])
    params["calibration_end"] = str(seasonal_model["calibration_end"])
    params["calibration_years"] = calibration_years
    params["price_floor"] = price_floor
    params["price_cap"] = price_cap

    return price_paths, future_datetimes, params
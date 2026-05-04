from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models import (
    simulate_gas_prices,
    simulate_generator_profit,
    price_asian_put,
)

from price_model import simulate_seasonal_price_paths

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

STRIKE_PERCENTILES = [25, 50, 60, 75]
RISK_FREE_RATE = 0.04


def risk_metrics(x):
    x = np.asarray(x)

    var5 = np.percentile(x, 5)
    es5 = x[x <= var5].mean()

    return {
        "mean": x.mean(),
        "std": x.std(ddof=1),
        "var5": var5,
        "es5": es5,
        "prob_loss": (x < 0).mean(),
    }


def optimize_forward_hedge(
    unhedged_profit,
    payoff_per_mwh,
    premium_per_mwh,
    base_notional,
):
    unhedged_profit = np.asarray(unhedged_profit)
    payoff_per_mwh = np.asarray(payoff_per_mwh)

    def objective(h):
        hedged = (
            unhedged_profit
            + h * base_notional * payoff_per_mwh
            - h * base_notional * premium_per_mwh
        )

        var5 = np.percentile(hedged, 5)
        es5 = hedged[hedged <= var5].mean()

        return -es5

    result = minimize_scalar(objective, bounds=(0, 1.0), method="bounded")
    return result.x


def summarize_forward_results(out):
    summary = (
        out.groupby("strike_percentile")
        .agg(
            months=("forward_month", "count"),
            avg_strike=("strike", "mean"),
            avg_premium_per_mwh=("premium_per_mwh", "mean"),
            avg_hedge_ratio=("optimal_hedge_ratio", "mean"),
            avg_mean_profit_change=("mean_profit_change", "mean"),
            avg_var5_improvement=("var5_improvement", "mean"),
            avg_es5_improvement=("es5_improvement", "mean"),
            avg_variance_reduction=("variance_reduction", "mean"),
        )
        .reset_index()
    )

    return summary

def apply_scarcity_month_shocks(
    price_paths,
    hours_per_month=720,
    scarcity_prob=0.002,
    shock_mean=2.5,
    shock_sigma=0.6,
    seed=404,
    price_floor=-250,
    price_cap=5000,
):
    """
    Apply rare ERCOT-style scarcity shocks to monthly blocks.

    The baseline price model captures normal seasonal and mean-reverting behavior.
    This function overlays rare scarcity regimes that increase prices for an
    entire simulated month.

    Parameters
    ----------
    price_paths : np.ndarray
        Simulated hourly price paths with shape (n_paths, n_hours).
    hours_per_month : int
        Number of hours in each modeled month.
    scarcity_prob : float
        Probability that a given path-month experiences scarcity.
    shock_mean : float
        Mean parameter of lognormal scarcity multiplier.
    shock_sigma : float
        Standard deviation parameter of lognormal scarcity multiplier.
    seed : int
        Random seed.
    price_floor : float
        Lower simulated price bound.
    price_cap : float
        Upper simulated price bound.

    Returns
    -------
    np.ndarray
        Shock-adjusted price paths.
    """
    rng = np.random.default_rng(seed)

    shocked = price_paths.copy()
    n_paths, total_hours = shocked.shape
    n_months = total_hours // hours_per_month

    for m in range(n_months):
        start = m * hours_per_month
        end = (m + 1) * hours_per_month

        scarcity_flag = rng.binomial(
            n=1,
            p=scarcity_prob,
            size=n_paths,
        )

        shock_multiplier = rng.lognormal(
            mean=shock_mean,
            sigma=shock_sigma,
            size=n_paths,
        )

        multiplier = 1.0 + scarcity_flag * shock_multiplier

        shocked[:, start:end] = (
            shocked[:, start:end] * multiplier.reshape(-1, 1)
        )

    return np.clip(shocked, price_floor, price_cap)


def main():
    config_path = PROJECT_ROOT / "asset_config.json"

    if not config_path.exists():
        raise FileNotFoundError("Missing asset_config.json")

    config = json.loads(config_path.read_text())
    asset = config["asset"]
    gm = config["generator_model"]

    data_path = DATA / "hb_north_with_gas.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "Missing data/processed/hb_north_with_gas.csv. "
            "Run: python src/process_data.py"
        )

    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["gas_price"] = pd.to_numeric(df["gas_price"], errors="coerce")

    df = df.dropna(subset=["datetime", "price", "gas_price"])
    df = df.sort_values("datetime")

    latest_power_datetime = df["datetime"].max()
    p0 = float(df["price"].iloc[-1])
    gas0 = float(df["gas_price"].iloc[-1])

    n_months = 24
    hours_per_month = 30 * 24
    total_hours = n_months * hours_per_month
    n_paths = 10000

    print(f"Historical data start: {df['datetime'].min()}")
    print(f"Historical data end:   {df['datetime'].max()}")
    print(f"Starting HB_NORTH:     ${p0:.2f}/MWh")
    print(f"Starting gas price:    ${gas0:.2f}/MMBtu")
    print(f"Simulating {n_paths:,} paths for {n_months} months...")

    price_paths, future_datetimes, price_params = simulate_seasonal_price_paths(
        df=df[["datetime", "price"]],
        start_datetime=latest_power_datetime + pd.Timedelta(hours=1),
        n_hours=total_hours,
        n_paths=n_paths,
        seed=2028,
    )

    price_paths = apply_scarcity_month_shocks(
    price_paths=price_paths,
    hours_per_month=hours_per_month,
    scarcity_prob=0.001,
    shock_mean=1.5,
    shock_sigma=0.6,
    seed=404,
    )

    np.save("outputs/sim_price_paths.npy", price_paths)

    gas_paths = simulate_gas_prices(
        gas0=gas0,
        n_hours=total_hours,
        n_paths=n_paths,
        seed=2029,
    )

    rows = []

    for m in range(n_months):
        start = m * hours_per_month
        end = (m + 1) * hours_per_month

        pp = price_paths[:, start:end]
        gp = gas_paths[:, start:end]

        gen = simulate_generator_profit(
            price_paths=pp,
            gas_paths=gp,
            capacity_mw=asset["net_summer_capacity_mw"],
            heat_rate=gm["heat_rate_mmbtu_per_mwh"],
            variable_om=gm["variable_om_usd_per_mwh"],
            availability_mean=gm["availability_mean"],
            availability_std=gm["availability_std"],
            min_availability=gm["min_availability"],
            max_availability=gm["max_availability"],
            dispatch_width=gm["dispatch_smoothing_width_usd_per_mwh"],
            capacity_factor_cap=gm["capacity_factor_cap"],
            seed=3000 + m,
        )

        unhedged = gen["monthly_profit"]
        base_notional = np.mean(gen["monthly_generation_mwh"])
        avg_sim_price = pp.mean(axis=1)

        for strike_pct in STRIKE_PERCENTILES:
            strike = np.percentile(avg_sim_price, strike_pct)

            option = price_asian_put(
                pp,
                strike=strike,
                annual_rate=RISK_FREE_RATE,
                hours=pp.shape[1],
            )

            premium = option["premium_per_mwh"]
            payoff = option["payoff_per_mwh"]

            h_star = optimize_forward_hedge(
                unhedged_profit=unhedged,
                payoff_per_mwh=payoff,
                premium_per_mwh=premium,
                base_notional=base_notional,
            )

            hedged = (
                unhedged
                + h_star * base_notional * payoff
                - h_star * base_notional * premium
            )

            u = risk_metrics(unhedged)
            h = risk_metrics(hedged)

            rows.append({
                "forward_month": m + 1,
                "strike_percentile": strike_pct,
                "strike": strike,
                "premium_per_mwh": premium,
                "optimal_hedge_ratio": h_star,
                "base_notional_mwh": base_notional,
                "mean_unhedged": u["mean"],
                "mean_hedged": h["mean"],
                "mean_profit_change": h["mean"] - u["mean"],
                "std_unhedged": u["std"],
                "std_hedged": h["std"],
                "var5_unhedged": u["var5"],
                "var5_hedged": h["var5"],
                "var5_improvement": h["var5"] - u["var5"],
                "es5_unhedged": u["es5"],
                "es5_hedged": h["es5"],
                "es5_improvement": h["es5"] - u["es5"],
                "prob_loss_unhedged": u["prob_loss"],
                "prob_loss_hedged": h["prob_loss"],
                "variance_reduction": 1 - np.var(hedged) / np.var(unhedged),
            })

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("Forward simulation produced no rows.")

    out_path = OUTPUTS / "two_year_forward_hedge_plan.csv"
    out.to_csv(out_path, index=False)

    summary = summarize_forward_results(out)
    summary_path = OUTPUTS / "two_year_forward_hedge_plan_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(out.head())
    print("\nSummary by strike percentile:")
    print(summary)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
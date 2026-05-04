from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from models import sigmoid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

CAPACITY_MW = 1770
HEAT_RATE = 7.5
VOM = 3.0
AVAILABILITY = 0.94
CAPACITY_FACTOR_CAP = 0.90
DISPATCH_WIDTH = 5.0
RISK_FREE_RATE = 0.04

STRIKE_QUANTILES = [0.25, 0.50, 0.60, 0.75]


def compute_generator_profit(df):
    df = df.copy()

    df["marginal_cost"] = HEAT_RATE * df["gas_price"] + VOM

    df["dispatch_mw"] = (
        CAPACITY_MW
        * AVAILABILITY
        * CAPACITY_FACTOR_CAP
        * sigmoid((df["price"] - df["marginal_cost"]) / DISPATCH_WIDTH)
    )

    df["hourly_profit"] = df["dispatch_mw"] * (
        df["price"] - df["marginal_cost"]
    )

    return df


def bootstrap_asian_put_premium(
    history_prices,
    strike,
    n_hours,
    n_sims=5000,
    seed=42,
):
    rng = np.random.default_rng(seed)
    hist = history_prices.dropna().values

    if len(hist) == 0:
        return 0.0, np.zeros(n_sims)

    sims = rng.choice(hist, size=(n_sims, n_hours), replace=True)
    avg_prices = sims.mean(axis=1)

    payoff = np.maximum(strike - avg_prices, 0)

    tau = n_hours / (365 * 24)
    discount = np.exp(-RISK_FREE_RATE * tau)

    premium = discount * payoff.mean()

    return premium, payoff


def optimize_hedge_ratio(
    unhedged_profit_samples,
    payoff_samples,
    premium_per_mwh,
    notional_mwh,
):
    unhedged_profit_samples = np.asarray(unhedged_profit_samples)
    payoff_samples = np.asarray(payoff_samples)

    def objective(h):
        hedged = (
            unhedged_profit_samples
            + h * notional_mwh * payoff_samples
            - h * notional_mwh * premium_per_mwh
        )

        var5 = np.percentile(hedged, 5)
        es5 = hedged[hedged <= var5].mean()

        return -es5

    result = minimize_scalar(objective, bounds=(0, 1.0), method="bounded")
    return result.x


def summarize_monthly_metrics(out):
    summary = (
        out.groupby("strike_quantile")
        .agg(
            months=("month", "count"),
            avg_strike=("strike", "mean"),
            avg_premium_per_mwh=("premium_per_mwh", "mean"),
            avg_hedge_ratio=("optimal_hedge_ratio", "mean"),
            avg_unhedged_profit=("unhedged_profit", "mean"),
            avg_hedged_profit=("hedged_profit", "mean"),
            avg_profit_change=("profit_change", "mean"),
            total_profit_change=("profit_change", "sum"),
            avg_premium_paid=("premium_paid", "mean"),
            avg_hedge_payoff=("hedge_payoff", "mean"),
        )
        .reset_index()
    )

    return summary


def main():
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

    df = compute_generator_profit(df)
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    cutoff = df["datetime"].max() - pd.DateOffset(years=5)
    df5 = df[df["datetime"] >= cutoff].copy()

    # Drop incomplete final month
    last_complete_month = (
        df["datetime"].max().to_period("M") - 1
    ).strftime("%Y-%m")

    df5 = df5[df5["month"] <= last_complete_month].copy()
    months = sorted(df5["month"].unique())

    rows = []

    for month in months:
        month_df = df5[df5["month"] == month]
        start = month_df["datetime"].min()

        hist_start = start - pd.DateOffset(years=2)
        hist = df[
            (df["datetime"] >= hist_start)
            & (df["datetime"] < start)
        ].copy()

        print(
            month,
            "month rows:", len(month_df),
            "hist rows:", len(hist),
            "start:", start,
        )

        if len(hist) < 24 * 30:
            print(f"Skipping {month}: only {len(hist)} trailing hourly observations")
            continue
        n_hours = len(month_df)

        avg_price = month_df["price"].mean()
        generation_mwh = month_df["dispatch_mw"].sum()
        unhedged_profit = month_df["hourly_profit"].sum()

        trailing_monthly_avg = hist.groupby("month")["price"].mean()
        hist_monthly_profit = hist.groupby("month")["hourly_profit"].sum().values

        if len(trailing_monthly_avg) < 6 or len(hist_monthly_profit) == 0:
            continue

        for strike_q in STRIKE_QUANTILES:
            strike = trailing_monthly_avg.quantile(strike_q)

            premium_per_mwh, payoff_samples = bootstrap_asian_put_premium(
                hist["price"],
                strike=strike,
                n_hours=n_hours,
                n_sims=5000,
                seed=123,
            )

            min_len = min(len(hist_monthly_profit), len(payoff_samples))

            unhedged_samples = np.random.default_rng(999).choice(
                hist_monthly_profit,
                size=min_len,
                replace=True,
            )

            payoff_for_opt = payoff_samples[:min_len]
            notional_mwh = generation_mwh

            hedge_ratio = optimize_hedge_ratio(
                unhedged_profit_samples=unhedged_samples,
                payoff_samples=payoff_for_opt,
                premium_per_mwh=premium_per_mwh,
                notional_mwh=notional_mwh,
            )

            realized_payoff_per_mwh = max(strike - avg_price, 0)
            premium_paid = hedge_ratio * notional_mwh * premium_per_mwh
            hedge_payoff = hedge_ratio * notional_mwh * realized_payoff_per_mwh

            hedged_profit = unhedged_profit + hedge_payoff - premium_paid

            rows.append({
                "month": month,
                "strike_quantile": strike_q,
                "avg_price": avg_price,
                "strike": strike,
                "premium_per_mwh": premium_per_mwh,
                "generation_mwh": generation_mwh,
                "notional_mwh": notional_mwh,
                "optimal_hedge_ratio": hedge_ratio,
                "unhedged_profit": unhedged_profit,
                "realized_payoff_per_mwh": realized_payoff_per_mwh,
                "hedge_payoff": hedge_payoff,
                "premium_paid": premium_paid,
                "hedged_profit": hedged_profit,
                "profit_change": hedged_profit - unhedged_profit,
            })

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("Backtest produced no rows. Check processed data coverage.")

    out_path = OUTPUTS / "five_year_historical_backtest.csv"
    out.to_csv(out_path, index=False)

    summary = summarize_monthly_metrics(out)
    summary_path = OUTPUTS / "five_year_historical_backtest_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(out.tail())
    print("\nSummary by strike quantile:")
    print(summary)
    print(f"\nSaved: {out_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
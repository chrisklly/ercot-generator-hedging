from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_hb_north_prices
from models import (
    add_time_features,
    estimate_mean_reverting_jump_model,
    simulate_hourly_prices,
    simulate_gas_prices,
    simulate_generator_profit,
    hedge_analysis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def load_latest_gas_price():
    gas_path = PROJECT_ROOT / "data" / "raw" / "gas_prices.csv"

    if not gas_path.exists():
        raise FileNotFoundError(
            "Missing data/raw/gas_prices.csv. "
            "Run: python src/download_gas.py"
        )

    gas = pd.read_csv(gas_path)
    gas["date"] = pd.to_datetime(gas["date"])
    gas["gas_price"] = pd.to_numeric(gas["gas_price"], errors="coerce")
    gas = gas.dropna().sort_values("date")

    latest_row = gas.iloc[-1]

    print(
        f"Using latest gas price: "
        f"${latest_row['gas_price']:.2f}/MMBtu on {latest_row['date'].date()}"
    )

    return float(latest_row["gas_price"])


def main():
    config = json.loads((PROJECT_ROOT / "asset_config.json").read_text())
    asset = config["asset"]
    gm = config["generator_model"]
    hedge_cfg = config["hedge"]

    # ------------------------------------------------------------
    # 1. Load historical ERCOT HB_NORTH prices
    # ------------------------------------------------------------
    prices = load_hb_north_prices(PROJECT_ROOT)
    prices = add_time_features(prices)

    if prices.empty:
        raise ValueError("No HB_NORTH price data loaded.")

    print(f"Loaded {len(prices):,} historical HB_NORTH price observations.")
    print(f"Price history starts: {prices['datetime'].min()}")
    print(f"Price history ends:   {prices['datetime'].max()}")

    # ------------------------------------------------------------
    # 2. Load historical gas data and set forward starting gas price
    # ------------------------------------------------------------
    latest_gas_price = load_latest_gas_price()

    # ------------------------------------------------------------
    # 3. Estimate historical power-price model
    # ------------------------------------------------------------
    model_params = estimate_mean_reverting_jump_model(prices["price"])

    p0 = float(prices["price"].iloc[-1])

    print("\nEstimated HB_NORTH price model:")
    for k, v in model_params.items():
        if k != "residuals":
            print(f"  {k}: {v}")

    # ------------------------------------------------------------
    # 4. Forward simulation settings
    # ------------------------------------------------------------
    n_hours = 30 * 24
    n_paths = 10_000

    # ------------------------------------------------------------
    # 5. Simulate forward HB_NORTH price paths
    # ------------------------------------------------------------
    price_paths = simulate_hourly_prices(
        p0=p0,
        params=model_params,
        n_hours=n_hours,
        n_paths=n_paths,
        seed=2026,
    )

    # ------------------------------------------------------------
    # 6. Simulate forward gas-price paths
    # ------------------------------------------------------------
    gas_paths = simulate_gas_prices(
        gas0=latest_gas_price,
        n_hours=n_hours,
        n_paths=n_paths,
        seed=2027,
    )

    # ------------------------------------------------------------
    # 7. Simulate generator dispatch and profit
    # ------------------------------------------------------------
    gen = simulate_generator_profit(
        price_paths=price_paths,
        gas_paths=gas_paths,
        capacity_mw=asset["net_summer_capacity_mw"],
        heat_rate=gm["heat_rate_mmbtu_per_mwh"],
        variable_om=gm["variable_om_usd_per_mwh"],
        availability_mean=gm["availability_mean"],
        availability_std=gm["availability_std"],
        min_availability=gm["min_availability"],
        max_availability=gm["max_availability"],
        dispatch_width=gm["dispatch_smoothing_width_usd_per_mwh"],
        capacity_factor_cap=gm["capacity_factor_cap"],
    )

    # ------------------------------------------------------------
    # 8. Define Asian put strike levels from historical monthly prices
    # ------------------------------------------------------------
    monthly_avg_hist = prices.groupby("month")["price"].mean()

    strikes = [
        float(monthly_avg_hist.quantile(0.10)),
        float(monthly_avg_hist.quantile(0.25)),
        float(monthly_avg_hist.quantile(0.50)),
    ]

    print("\nAsian put strike candidates:")
    for s in strikes:
        print(f"  ${s:.2f}/MWh")

    # ------------------------------------------------------------
    # 9. Hedge-effectiveness grid search
    # ------------------------------------------------------------
    results = []
    full_results = {}

    for strike in strikes:
        for hedge_ratio in hedge_cfg["hedge_ratio_grid"]:
            res = hedge_analysis(
                unhedged_profit=gen["monthly_profit"],
                price_paths=price_paths,
                generation_mwh=gen["monthly_generation_mwh"],
                strike=strike,
                hedge_ratio=hedge_ratio,
                annual_rate=hedge_cfg["annual_risk_free_rate"],
            )

            row = {k: v for k, v in res.items() if k != "hedged_profit"}
            results.append(row)
            full_results[(float(strike), float(hedge_ratio))] = res

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUTS / "hedge_effectiveness_results.csv", index=False)

    # ------------------------------------------------------------
    # 10. Select a best hedge case
    # ------------------------------------------------------------
    baseline_mean = results_df["mean_unhedged"].iloc[0]

    feasible = results_df[
        results_df["mean_hedged"] >= baseline_mean * 0.95
    ].copy()

    if len(feasible):
        best = feasible.sort_values(
            ["es5_improvement", "variance_reduction"],
            ascending=False,
        ).iloc[0]
    else:
        best = results_df.sort_values(
            ["es5_improvement", "variance_reduction"],
            ascending=False,
        ).iloc[0]

    # ------------------------------------------------------------
    # 11. Save plots
    # ------------------------------------------------------------
    plt.figure(figsize=(11, 5))
    prices.set_index("datetime")["price"].plot()
    plt.title("Historical ERCOT North Hub Price Input")
    plt.ylabel("$/MWh")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "historical_hb_north_prices.png", dpi=150)
    plt.close()

    best_key = (float(best["strike"]), float(best["hedge_ratio"]))
    best_hedged = full_results[best_key]["hedged_profit"]

    plt.figure(figsize=(10, 5))
    plt.hist(
        gen["monthly_profit"],
        bins=80,
        alpha=0.55,
        label="Unhedged",
    )
    plt.hist(
        best_hedged,
        bins=80,
        alpha=0.55,
        label="Best hedged case",
    )
    plt.title("Forward-Simulated Monthly Profit Distribution")
    plt.xlabel("Monthly profit ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "profit_distribution_unhedged_vs_hedged.png", dpi=150)
    plt.close()

    summary = (
        "# ERCOT Forney Asian Put Hedge Analysis - Run Summary\n\n"
        f"Asset: {asset['name']}\n"
        f"Capacity: {asset['net_summer_capacity_mw']:.0f} MW\n\n"
        f"Price Index: {asset['price_proxy']}\n\n"
        f"Data Range: {prices['datetime'].min()} to {prices['datetime'].max()}\n"
        f"Observations: {len(prices):,}\n\n"
        f"Starting Power Price: ${p0:.2f}\n"
        f"Starting Gas Price: ${latest_gas_price:.2f}\n\n"
        f"Best Hedge:\n"
        f"  Strike: ${best['strike']:.2f}\n"
        f"  Hedge Ratio: {best['hedge_ratio']:.2f}\n\n"
        f"Risk Metrics:\n"
        f"  Mean Unhedged: ${best['mean_unhedged']:,.0f}\n"
        f"  Mean Hedged: ${best['mean_hedged']:,.0f}\n"
        f"  VaR (5%) Unhedged: ${best['var5_unhedged']:,.0f}\n"
        f"  VaR (5%) Hedged: ${best['var5_hedged']:,.0f}\n"
        f"  Variance Reduction: {100 * best['variance_reduction']:.1f}%\n"
    )

    (OUTPUTS / "run_summary.md").write_text(summary)
    print("\nSaved run_summary.md")

if __name__ == "__main__":
    main()
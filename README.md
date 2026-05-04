# ERCOT Forney Energy Center Asian Put Hedge Project

## Project 

How effective are monthly average-price Asian put options on ERCOT North Hub (HB_NORTH) as a hedge for downside revenue risk at Forney Energy Center, a large natural-gas combined-cycle generator in ERCOT?


## Asset

Selected asset: **Forney Energy Center**

Initial asset parameters are stored in `asset_config.json`.

## Public data inputs

1. ERCOT historical Day-Ahead Market Load Zone and Hub Prices  
   Use HB_NORTH as the hedge index / power price proxy.  
   Source: https://www.ercot.com/mp/data-products/data-product-details?id=NP4-180-ER

2. ERCOT market price overview  
   Source: https://www.ercot.com/mktinfo/prices

3. EIA Form 860M generator inventory  
   Source: https://www.eia.gov/electricity/data/eia860m/

4. Optional: gas price input  
   Use Henry Hub, Waha, or another Texas-relevant gas price. For a first version, use a constant gas price and sensitivity cases.

## Data setup

Place ERCOT historical DAM hub/load-zone price files in:

`data/raw/ercot_dam_prices/`

Expected columns, after cleaning:
- date or delivery_date
- hour_ending
- HB_NORTH

You can also provide a single CSV at:

`data/raw/hb_north_prices.csv`

with columns:
- datetime
- price

If no price CSV is available, the starter script can run a synthetic demonstration dataset so the pipeline still executes.

## Main commands

From this folder:

```bash
pip install -r requirements.txt
python src/run_analysis.py
```

Outputs are written to:

`outputs/`

## Core model

Hourly generator margin:

```text
margin_t = power_price_t - marginal_cost_t
marginal_cost_t = heat_rate * gas_price_t + variable_om
```

Smooth economic dispatch:

```text
q_t = capacity * availability_t * sigmoid((price_t - cost_t) / width)
```

Monthly unhedged profit:

```text
sum_t q_t * (price_t - cost_t)
```

Asian put payoff:

```text
hedge_notional_mwh * max(strike - average_HB_NORTH_price, 0)
```

Monthly hedged profit:

```text
unhedged_profit + put_payoff - premium_paid
```

## Hedge-effectiveness metrics

The final analysis reports:
- mean profit
- profit standard deviation
- 5% VaR
- 5% expected shortfall
- probability of loss
- variance reduction
- downside-risk improvement
- break-even premium
- optimal hedge ratio grid search

## Recommended final report structure

1. Introduction and project motivation
2. ERCOT market and Forney asset description
3. Data sources and cleaning
4. Generator economics model
5. Asian put valuation methodology
6. Monte Carlo simulation design
7. Hedge-effectiveness results
8. Sensitivity analysis
9. Limitations and extensions
10. Conclusion

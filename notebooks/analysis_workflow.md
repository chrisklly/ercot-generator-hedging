# Analysis Notebook Outline

This markdown file is a notebook guide. You can copy these cells into a `.ipynb` notebook.

## 1. Imports

```python
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_hb_north_prices
from src.models import *
```

## 2. Load ERCOT North Hub data

```python
project_root = Path(".")
prices = load_hb_north_prices(project_root)
prices.head()
```

## 3. Estimate price model

```python
prices = add_time_features(prices)
params = estimate_mean_reverting_jump_model(prices["price"])
params
```

## 4. Simulate HB_NORTH paths

```python
paths = simulate_hourly_prices(
    p0=prices["price"].iloc[-1],
    params=params,
    n_hours=30*24,
    n_paths=10000,
)
```

## 5. Simulate generator economics

```python
gas = simulate_gas_prices(gas0=3.25, n_hours=30*24, n_paths=10000)
gen = simulate_generator_profit(
    price_paths=paths,
    gas_paths=gas,
    capacity_mw=1770,
)
```

## 6. Evaluate Asian put hedge

```python
res = hedge_analysis(
    unhedged_profit=gen["monthly_profit"],
    price_paths=paths,
    generation_mwh=gen["monthly_generation_mwh"],
    strike=35,
    hedge_ratio=0.5,
)
res
```

## 7. Grid search strikes and hedge ratios

```python
strikes = [30, 35, 40, 45]
hedge_ratios = [0, 0.25, 0.5, 0.75, 1.0]

rows = []
for k in strikes:
    for h in hedge_ratios:
        r = hedge_analysis(gen["monthly_profit"], paths, gen["monthly_generation_mwh"], k, h)
        rows.append({key: val for key, val in r.items() if key != "hedged_profit"})

pd.DataFrame(rows)
```

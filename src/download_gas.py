from pathlib import Path
from io import StringIO

import pandas as pd
import requests
import certifi

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP"

response = requests.get(url, timeout=30, verify=certifi.where())
response.raise_for_status()

gas = pd.read_csv(StringIO(response.text))
gas.columns = ["date", "gas_price"]

gas["date"] = pd.to_datetime(gas["date"])
gas["gas_price"] = pd.to_numeric(gas["gas_price"], errors="coerce")
gas = gas.dropna()

gas.to_csv(RAW / "gas_prices.csv", index=False)

print(f"Saved {len(gas):,} rows to data/raw/gas_prices.csv")
print(gas.tail())
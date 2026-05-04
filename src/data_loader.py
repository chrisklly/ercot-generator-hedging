from pathlib import Path
import numpy as np
import pandas as pd

def load_hb_north_prices(project_root: Path) -> pd.DataFrame:
    """
    Load ERCOT HB_NORTH hourly prices.

    Preferred input:
        data/raw/hb_north_prices.csv
    with:
        datetime, price

    Alternative:
        data/raw/ercot_dam_prices/*.csv
    with ERCOT-style columns including Oper Day, Hour Ending, HB_NORTH.

    Returns:
        DataFrame with columns datetime, price
    """
    direct = project_root / "data" / "raw" / "hb_north_prices.csv"
    if direct.exists():
        df = pd.read_csv(direct)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        return df[["datetime", "price"]].dropna().sort_values("datetime")

    folder = project_root / "data" / "raw" / "ercot_dam_prices"
    if folder.exists():
        files = sorted(folder.glob("*.csv"))
        frames = []
        for f in files:
            tmp = pd.read_csv(f)
            cols = {c.lower().strip(): c for c in tmp.columns}
            date_col = cols.get("oper day") or cols.get("delivery date") or cols.get("date")
            hour_col = cols.get("hour ending") or cols.get("hour")
            price_col = cols.get("hb_north")
            if date_col and hour_col and price_col:
                h = pd.to_numeric(tmp[hour_col], errors="coerce")
                # ERCOT hour ending 24 belongs to the 23:00-24:00 interval.
                dt = pd.to_datetime(tmp[date_col]) + pd.to_timedelta(h.fillna(1).astype(int) - 1, unit="h")
                frames.append(pd.DataFrame({
                    "datetime": dt,
                    "price": pd.to_numeric(tmp[price_col], errors="coerce")
                }))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            return df.dropna().sort_values("datetime")

    return make_synthetic_hb_north_prices()

def make_synthetic_hb_north_prices(seed: int = 42, years: int = 3) -> pd.DataFrame:
    """
    Synthetic fallback so the code runs before ERCOT data is downloaded.
    Replace this with real ERCOT HB_NORTH data for the final project.
    """
    rng = np.random.default_rng(seed)
    n = years * 365 * 24
    dt = pd.date_range("2023-01-01", periods=n, freq="h")

    hour = dt.hour.values
    dayofyear = dt.dayofyear.values

    daily_shape = 8 * np.sin(2 * np.pi * (hour - 15) / 24)
    seasonal = 12 * np.sin(2 * np.pi * (dayofyear - 170) / 365)
    base = 38 + daily_shape + seasonal

    noise = rng.normal(0, 8, n)
    spike_flag = rng.binomial(1, 0.012, n)
    spikes = spike_flag * rng.lognormal(mean=4.0, sigma=0.65, size=n)
    negative_flag = rng.binomial(1, 0.006, n)
    negatives = negative_flag * rng.normal(35, 10, n)

    price = base + noise + spikes - negatives
    return pd.DataFrame({"datetime": dt, "price": price})

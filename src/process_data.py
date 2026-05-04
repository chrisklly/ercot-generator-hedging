from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def parse_hour_ending(x):
    if pd.isna(x):
        return None

    s = str(x).strip()

    if ":" in s:
        s = s.split(":")[0]

    try:
        h = int(float(s))
        if 1 <= h <= 24:
            return h
    except ValueError:
        return None

    return None


def clean_price(x):
    if pd.isna(x):
        return None
    s = str(x).replace("$", "").replace(",", "").strip()

    try:
        return float(s)
    except ValueError:
        return None


def read_ercot_file(path):
    """
    Reads ERCOT DAM Load Zone and Hub Prices files across all sheets.
    Expected columns:
    Delivery Date, Hour Ending, Settlement Point, Settlement Point Price
    """

    def parse_one_sheet(df, sheet_name=None):
        df.columns = [str(c).strip() for c in df.columns]

        required = {
            "Delivery Date",
            "Hour Ending",
            "Settlement Point",
            "Settlement Point Price",
        }

        missing = required - set(df.columns)
        if missing:
            return pd.DataFrame(columns=["datetime", "price"])

        df = df[
            df["Settlement Point"].astype(str).str.strip().str.upper() == "HB_NORTH"
        ].copy()

        df["hour"] = df["Hour Ending"].apply(parse_hour_ending)
        df["price"] = df["Settlement Point Price"].apply(clean_price)
        df["date"] = pd.to_datetime(df["Delivery Date"], errors="coerce")

        df = df.dropna(subset=["date", "hour", "price"])

        df["datetime"] = df["date"] + pd.to_timedelta(
            df["hour"].astype(int) - 1,
            unit="h",
        )

        return df[["datetime", "price"]]

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        return parse_one_sheet(df)

    sheets = pd.read_excel(path, sheet_name=None, header=0, engine="openpyxl")
    frames = []

    for sheet_name, sheet_df in sheets.items():
        parsed = parse_one_sheet(sheet_df, sheet_name=sheet_name)

        if not parsed.empty:
            frames.append(parsed)

    if not frames:
        print(f"No usable sheets found in {path.name}")
        return pd.DataFrame(columns=["datetime", "price"])

    return pd.concat(frames, ignore_index=True)


def process_ercot_prices():
    folder = RAW / "ercot_dam_prices"

    files = [
    f for f in (list(folder.rglob("*.xlsx")) + list(folder.rglob("*.csv")))
    if not f.name.startswith("~$")
    ]

    if not files:
        raise FileNotFoundError(
            "No ERCOT files found under data/raw/ercot_dam_prices"
        )

    dfs = []

    for f in files:
        print(f"Reading {f}")
        df = read_ercot_file(f)

        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise ValueError("No HB_NORTH rows found in ERCOT files.")

    prices = (
        pd.concat(dfs, ignore_index=True)
        .dropna()
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
    )

    prices.to_csv(PROCESSED / "hb_north_clean.csv", index=False)

    print(f"Saved hb_north_clean.csv with {len(prices):,} rows")
    print(f"Start: {prices['datetime'].min()}")
    print(f"End:   {prices['datetime'].max()}")

    return prices


def process_gas():
    gas_path = RAW / "gas_prices.csv"

    gas = pd.read_csv(gas_path)
    gas.columns = ["date", "gas_price"]

    gas["date"] = pd.to_datetime(gas["date"])
    gas["gas_price"] = pd.to_numeric(gas["gas_price"], errors="coerce")
    gas = gas.dropna().sort_values("date")

    hourly = (
        gas.set_index("date")["gas_price"]
        .resample("h")
        .ffill()
        .rename("gas_price")
        .reset_index()
        .rename(columns={"date": "datetime"})
    )

    hourly.to_csv(PROCESSED / "gas_hourly.csv", index=False)

    print(f"Saved gas_hourly.csv with {len(hourly):,} rows")

    return hourly


def merge_data(prices, gas_hourly):
    df = prices.merge(
        gas_hourly,
        on="datetime",
        how="left",
    )

    df["gas_price"] = df["gas_price"].ffill().bfill()
    df["marginal_cost"] = 7.0 * df["gas_price"] + 3.0

    df.to_csv(PROCESSED / "hb_north_with_gas.csv", index=False)

    print(f"Saved hb_north_with_gas.csv with {len(df):,} rows")

    return df


def main():
    prices = process_ercot_prices()
    gas_hourly = process_gas()
    merge_data(prices, gas_hourly)


if __name__ == "__main__":
    main()
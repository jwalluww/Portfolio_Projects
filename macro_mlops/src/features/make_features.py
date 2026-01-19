import pandas as pd

RAW_DATA_PATH = "macro_mlops/data/raw/fred_data.csv"
PROCESSED_DATA_PATH = "macro_mlops/data/processed/fred_data.csv"

def make_features():
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    df = df.set_index("date").sort_index()

    # Forward fill gdp
    df["gdp"] = df["gdp"].ffill()

    # shift CPI to be the target for next month
    df["cpi_target"] = df["cpi"].shift(-1)

    # Lag all features by 1 month to prevent data leakage
    for col in df.columns:
        if col != "cpi_target":
            df[f"{col}_lag1"] = df[col].shift(1)

    # drop rows with missing values (created by shifting)
    df = df.dropna()
    
    # keep only lagged features & target
    feature_cols = [col for col in df.columns if col.endswith("_lag1")]
    df = df[["cpi_target"] + feature_cols]

    # save it!
    df.to_csv(PROCESSED_DATA_PATH, index=True)
    print(f"✅ Features saved to {PROCESSED_DATA_PATH}")
    print(df.head())

if __name__ == "__main__":
    make_features()
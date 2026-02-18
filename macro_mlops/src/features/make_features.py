import pandas as pd

RAW_DATA_PATH = "data/raw/fred_data.csv"
PROCESSED_DATA_PATH = "data/processed/fred_data.csv"

def make_features(raw_df) -> pd.DataFrame:
    df = raw_df.copy()
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

    print("✅ Features created successfully.")
    print(df.head())

    return df

def make_features_from_file_and_save():
    raw_df = pd.read_csv(RAW_DATA_PATH)
    df = make_features(raw_df)
    df.to_csv(PROCESSED_DATA_PATH, index=True)
    print(f"✅ Features saved to {PROCESSED_DATA_PATH}")
    print(df.head())
    return df

if __name__ == "__main__":
    make_features_from_file_and_save()
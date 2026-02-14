import os
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# dictionary of FRED columns to pull in
fred_cols = {
    "cpi": "CPIAUCSL",  # Consumer Price Index for All Urban Consumers: All Items
    "unemployment_rate": "UNRATE",  # Unemployment Rate
    "interest_rate": "FEDFUNDS",  # Effective Federal Funds Rate
    "oil_price": "WTISPLC",  # Crude Oil Prices: West Texas Intermediate (WTI)WTI Spot Oil Price
    "gdp": "GDP",  # Gross Domestic Product (quarterly)
    "m2_money": "M2SL",  # M2 Money Stock
}

def fetch_fred(fred_cols, start_date="2000-01-01"):
    """
    Fetch data from FRED for the specified columns and date range.

    Parameters:
    fred_cols (dict): Dictionary mapping column names to FRED series IDs.
    start_date (str): Start date for data retrieval in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: DataFrame containing the fetched data with dates as index.
    """
    df = pd.DataFrame()
    for col, id in fred_cols.items():
        print(f"Fetching {col} data from FRED...")
        raw_data = fred.get_series(id)
        data = raw_data[start_date:].to_frame(name=col)
        data.index.name = "date"
        df = pd.concat([df, data], axis=1)
    return df

if __name__ == "__main__":
    df = fetch_fred(fred_cols)
    df.to_csv(f"data/raw/fred_data.csv")
    print("Data fetched and saved to CSV.")
    print(df.head())
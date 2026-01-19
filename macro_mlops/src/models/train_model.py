import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import os
import json

DATA_PATH = "macro_mlops/data/processed/fred_data.csv"
MODEL_PATH = "macro_mlops/src/models/model.joblib"
METRICS_PATH = "macro_mlops/src/models/metrics.json"

def train_model():
    # Load the processed data
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values(by="date")

    # Time-based split (80/20)
    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df.drop(columns=["date", "cpi_target"])
    print(f"Training features: {X_train.columns.tolist()}")
    y_train = train_df["cpi_target"]
    X_test = test_df.drop(columns=["date", "cpi_target"])
    y_test = test_df["cpi_target"]

    # fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Eval metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
    
    print("✅ Model Performance:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Save model + metrics
    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n🎯 Model saved to: {MODEL_PATH}")
    print(f"📊 Metrics saved to: {METRICS_PATH}")

if __name__ == "__main__":
    train_model()
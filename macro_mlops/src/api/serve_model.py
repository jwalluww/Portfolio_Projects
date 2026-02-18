from fastapi import FastAPI, HTTPException
from features import make_features
from pydantic import BaseModel
import threading
import joblib
import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from datetime import datetime
# Load these from inside docker /app
from src.utils.db import SessionLocal, PredictionLog, start_retrain_run, finish_retrain_run
from src.models.train_model import train_model
from src.monitoring.monitor_drift import check_drift
from src.ingestion.fetch_data import fetch_fred

# =========================
# Constants & Config
# =========================

version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
VERSIONED_MODEL_PATH = f"src/models/model_{version}.joblib"
MODEL_CURRENT_PATH = "src/models/model_current.joblib"
MODEL_CANDIDATE_PATH = "src/models/model_candidate.joblib"
TRAINING_DATA_PATH = "src/data/training_data.csv"

RETRAIN_SECRET = os.getenv("RETRAIN_SECRET")
MAX_DRIFT_ROWS = 1000

# =========================
# App Init
# =========================

app = FastAPI(title="Inflation Predictor API")
RETRAINING = False  # in-memory lock

# =========================
# Schemas
# =========================

class InflationInput(BaseModel):
    cpi_lag1: float
    unemployment_rate_lag1: float
    interest_rate_lag1: float
    oil_price_lag1: float
    gdp_lag1: float
    m2_money_lag1: float

# =========================
# Helper Functions
# =========================

def load_model():
    """Load the currently active model (supports hot swapping)."""
    return joblib.load(MODEL_CURRENT_PATH)


def load_training_data():
    """Reference original training dataset for drift detection."""
    return pd.read_csv(TRAINING_DATA_PATH)


def load_recent_predictions(limit: int = MAX_DRIFT_ROWS):
    """Pull recent prediction logs from Postgres."""
    db = SessionLocal()
    records = (
        db.query(PredictionLog)
        .order_by(PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    db.close()

    return pd.DataFrame([r.__dict__ for r in records])


def should_retrain(reference_df, current_df):
    """Run drift detection and return decision + metrics."""
    drift_result = check_drift(reference_df, current_df)
    return drift_result["drift_detected"], drift_result


def run_retraining(run_id: int, drift_metrics: dict):
    """Retrain model and update retrain run status in DB."""
    global RETRAINING
    RETRAINING = True

    try:
        # 1. Fetch fresh FRED data
        raw_df = fetch_fred()

        # 2. Build features
        features_df = make_features(raw_df)

        # 3. Train candidate model
        candidate_metrics = train_model(
            features_df, output_path=MODEL_CANDIDATE_PATH
        )

        # 4. Evaluate current model
        current_model = load_model()
        current_metrics = evaluate_model(current_model, features_df)

        # 5 Decide whether to swap
        improvement = candidate_metrics["rmse"] < current_metrics["rmse"]

        if improvement:
            os.replace(MODEL_CANDIDATE_PATH, MODEL_CURRENT_PATH)
            status = "succeeded"
        else:
            status = "no_improvement"

        merged_metrics = {
            "drift": drift_metrics,
            "train": candidate_metrics,
            "current": current_metrics,
            "improvement": improvement
        }

        finish_retrain_run(run_id, status=status, metrics=merged_metrics)

    except Exception as e:
        finish_retrain_run(run_id, status="failed", metrics={"error": str(e)})
        raise
    finally:
        RETRAINING = False


def train_and_save_model(output_path: str):
    """Train model and save to disk."""
    df = load_training_data()
    model, metrics = train_model(df, output_path)
    joblib.dump(model, output_path)
    return metrics


def evaluate_model(model, df):
    """Evaluate model on new data."""
    X = df.drop(columns=['cpi_target'])
    y = df['cpi_target']
    preds = model.predict(X)
    return {
        "mae": mean_absolute_error(y, preds),
        "rmse": root_mean_squared_error(y, preds),
        "r2": r2_score(y, preds)
    }

# =========================
# Endpoints
# =========================

@app.get("/")
def root():
    return {"status": "Inflation prediction API running"}


@app.post("/predict")
def predict(input_data: InflationInput):
    model = load_model()

    features_df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(features_df)[0]

    # Log prediction
    db = SessionLocal()
    record = PredictionLog(
        **input_data.dict(),
        prediction=float(prediction)
    )
    db.add(record)
    db.commit()
    db.close()

    return {"predicted_inflation": float(prediction)}


@app.post("/retrain")
def retrain(secret: str):
    global RETRAINING

    # Auth
    if secret != RETRAIN_SECRET:
        raise HTTPException(status_code=403, detail="Unauthorized")

    # Concurrency guard
    if RETRAINING:
        return {"status": "retraining already in progress"}

    reference_df = load_training_data()
    current_df = load_recent_predictions()

    if current_df.empty:
        return {"status": "no recent predictions to evaluate for drift"}

    retrain_decision, drift_metrics = should_retrain(
        reference_df, current_df
    )

    # Create retrain run entry
    run_id = start_retrain_run()

    if retrain_decision:
        threading.Thread(target=run_retraining, args=(run_id, drift_metrics)).start()
        return {
            "status": "retraining started",
            "run_id": run_id,
            "drift_metrics": drift_metrics,
        }
    
    # if skipped, mark run as skipped
    finish_retrain_run(run_id, status="skipped", metrics={"drift": drift_metrics})

    return {
        "status": "retraining skipped",
        "run_id": run_id,
        "drift_metrics": drift_metrics,
    }


@app.get("/drift_status")
def drift_status():
    reference_df = load_training_data()
    current_df = load_recent_predictions()
    _, drift_metrics = should_retrain(reference_df, current_df)
    return drift_metrics
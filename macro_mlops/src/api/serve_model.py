from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
import joblib
import os
import pandas as pd

from src.utils.db import SessionLocal, PredictionLog, start_retrain_run, finish_retrain_run
from src.models.train_model import train_and_save_model
from macro_mlops.src.monitoring.monitor_drift import check_drift

# =========================
# Constants & Config
# =========================

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
    """Reference dataset for drift detection."""
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
    global RETRAINING
    RETRAINING = True
    try:
        train_metrics = train_and_save_model(output_path=MODEL_CANDIDATE_PATH)
        os.replace(MODEL_CANDIDATE_PATH, MODEL_CURRENT_PATH)

        merged = {
            "drift": drift_metrics,
            "train": train_metrics,
        }
        finish_retrain_run(run_id, status="succeeded", metrics=merged)

    except Exception as e:
        finish_retrain_run(run_id, status="failed", metrics={"error": str(e)})
        raise
    finally:
        RETRAINING = False

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

    retrain_decision, drift_metrics = should_retrain(
        reference_df, current_df
    )

    if retrain_decision:
        threading.Thread(target=run_retraining).start()
        return {
            "status": "retraining started",
            "drift_metrics": drift_metrics,
        }

    return {
        "status": "retraining skipped",
        "drift_metrics": drift_metrics,
    }


@app.get("/drift_status")
def drift_status():
    reference_df = load_training_data()
    current_df = load_recent_predictions()
    _, drift_metrics = should_retrain(reference_df, current_df)
    return drift_metrics
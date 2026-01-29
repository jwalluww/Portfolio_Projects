from fastapi import FastAPI, Depends, HTTPException # Return proper errors
import threading # run stuff in background
import time
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from src.utils.db import SessionLocal, PredictionLog # Importing the database session and model
from src.models.train_model import train_and_save_model  # Ensure the model is trained and saved

app = FastAPI(title="Inflation Predictor API")

# Load the model
model = joblib.load("src/models/model.joblib")

# Define expected input schema
class InflationInput(BaseModel):
    cpi_lag1: float
    unemployment_rate_lag1: float
    interest_rate_lag1: float
    oil_price_lag1: float
    gdp_lag1: float
    m2_money_lag1: float

@app.get("/")
def root():
    return {"message": "Inflation prediction API is up and running."}

def load_model():
    return joblib.load("src/models/model_current.joblib")

@app.post("/predict")
def predict(input_data: InflationInput):

    columns = [
        "cpi_lag1",
        "unemployment_rate_lag1",
        "interest_rate_lag1",
        "oil_price_lag1",
        "gdp_lag1",
        "m2_money_lag1"
    ]

    features_df = pd.DataFrame([input_data.dict()], columns=columns)
    model = load_model()
    prediction = model.predict(features_df)[0]

    print("🔥 PREDICT CALLED")
    print(input_data.dict())

    # Log scores to local DB for monitoring
    db = SessionLocal()
    record = PredictionLog(**{**input_data.dict(), 'prediction': float(prediction)})
    db.add(record)
    print("🔥 LOGGED TO DB")
    db.commit()
    db.close()

    return {"predicted_inflation": float(prediction)}

# We will use a simple global flag to indicate if retraining is in progress
RETRAINING = False

# Use a secret so not everyone can trigger retraining
@app.post("/retrain")
def retrain_model(secret: str):
    global RETRAINING

    # Simple secret check
    if secret != os.getenv("RETRAIN_SECRET"):
        return HTTPException(status_code=403, detail="Unauthorized")
    
    # Check if already retraining, avoid concurrent retraining
    if RETRAINING:
        return {"status": "Already running"}
    
    def retrain_job():
        global RETRAINING
        RETRAINING = True
        
        metrics = train_and_save_model(output_path="src/models/model_candidate.joblib")

        # Swap!
        os.replace("src/models/model_candidate.joblib", "src/models/model_current.joblib")

        RETRAINING = False
    
    # This will run in a separate thread to avoid blocking the API
    threading.Thread(target=retrain_job).start()

    return {"status": "Retraining started"}
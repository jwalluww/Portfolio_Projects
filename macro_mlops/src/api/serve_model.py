from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from scr.utils.db import SessionLocal, PredictionLog # Importing the database session and model

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
    prediction = model.predict(features_df)[0]

    # Log scores to local DB for monitoring
    db = SessionLocal()
    record = PredictionLog(**input_data.dict(), prediction=prediction)
    db.add(record)
    db.commit()
    db.close()

    return {"predicted_inflation": float(prediction)}
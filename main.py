from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Load trained XGBoost model
model = joblib.load("xgboost_appliance_model.pkl")

# Define FastAPI app
app = FastAPI()

# Pydantic model
class Features(BaseModel):
    T_out: float
    RH_out: float
    Visibility: float
    Tdewpoint: float
    hour: int
    day_of_week: int
    is_weekend: int
    Appliances_lag1: float
    Appliances_lag24: float
    Appliances_roll3: float
    Appliances_roll6: float

@app.get("/")
def read_root():
    return {"message": "Smart Home Energy Prediction API is running."}

@app.post("/predict")
def predict(data: Features):
    features = np.array([[v for v in data.dict().values()]])
    prediction = model.predict(features)[0]
    return {"predicted_usage": round(prediction, 2)}

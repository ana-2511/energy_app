# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model
model = joblib.load("xgboost_appliance_model.pkl")

# Define input schema
class InputData(BaseModel):
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

@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array([[
            data.T_out, data.RH_out, data.Visibility, data.Tdewpoint,
            data.hour, data.day_of_week, data.is_weekend,
            data.Appliances_lag1, data.Appliances_lag24,
            data.Appliances_roll3, data.Appliances_roll6
        ]])
        pred = model.predict(features)[0]
        return {"predicted_usage": float(pred)}
    except Exception as e:
        return {"error": str(e)}


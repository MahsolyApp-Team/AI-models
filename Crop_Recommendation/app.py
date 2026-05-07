from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rf_model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

app = FastAPI()

class CropInput(BaseModel):
    N: float = Field(..., ge=0, description="Nitrogen content in soil")
    P: float = Field(..., ge=0, description="Phosphorus content in soil")
    K: float = Field(..., ge=0, description="Potassium content in soil")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity in %")
    ph: float = Field(..., ge=0, le=14, description="pH value of soil")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")

class CropOutput(BaseModel):
    recommended_crop: str

def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df['NPK_ratio'] = df['N'] / (df['P'] + df['K'] + 1)
    df['total_nutrients'] = df['N'] + df['P'] + df['K']
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    df['rainfall_humidity_ratio'] = df['rainfall'] / (df['humidity'] + 1)
    return df[feature_names]  # enforce column order from training

@app.get("/")
def root():
    return {"message": "Crop Recommendation API is running. Visit /docs for the interactive UI."}


@app.post("/predict", response_model=CropOutput)
def predict(input_data: CropInput):
    try:
        raw = input_data.model_dump()
        df = engineer_features(raw)
        prediction = rf_model.predict(df)
        crop = label_encoder.inverse_transform(prediction)[0]
        return CropOutput(recommended_crop=crop)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
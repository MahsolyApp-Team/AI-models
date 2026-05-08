import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.pipeline import Pipeline
from contextlib import asynccontextmanager

# ─── MODEL LOADING ────────────────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

def load_artifact(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

# Globals populated at startup
rf_model     = None
scaler       = None
le_soil      = None
le_crop      = None
le_target    = None
column_names = None
pipeline     = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rf_model, scaler, le_soil, le_crop, le_target, column_names, pipeline
    rf_model     = load_artifact("rf_model.pkl")
    scaler       = load_artifact("scaler.pkl")
    le_soil      = load_artifact("le_soil.pkl")
    le_crop      = load_artifact("le_crop.pkl")
    le_target    = load_artifact("le_target.pkl")
    column_names = load_artifact("column_names.pkl")
    pipeline     = Pipeline([("scaler", scaler), ("rf_model", rf_model)])
    print("✅ Model artifacts loaded successfully.")
    yield

# ─── APP SETUP ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Fertilizer Recommendation API",
    description=(
        "Predicts the best fertilizer for a given set of soil, crop, "
        "and environmental conditions using a trained Random Forest classifier."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── SCHEMAS ──────────────────────────────────────────────────────────────────

class FertilizerRequest(BaseModel):
    Temparature: float = Field(..., example=26.0,   description="Temperature in °C")
    Humidity:    float = Field(..., example=52.0,   description="Relative humidity (%)")
    Moisture:    float = Field(..., example=38.0,   description="Soil moisture (%)")
    Soil_Type:   str   = Field(..., alias="Soil Type", example="Loamy",
                               description="Soil type as a string (e.g. Loamy, Sandy, Clayey)")
    Crop_Type:   str   = Field(..., alias="Crop Type", example="Wheat",
                               description="Crop type as a string (e.g. Wheat, Maize, Sugarcane)")
    Nitrogen:    float = Field(..., example=37.0,   description="Soil nitrogen (kg/ha)")
    Potassium:   float = Field(..., example=0.0,    description="Soil potassium (kg/ha)")
    Phosphorous: float = Field(..., example=0.0,    description="Soil phosphorous (kg/ha)")

    model_config = {"populate_by_name": True}

    @field_validator("Temparature", "Humidity", "Moisture",
                     "Nitrogen", "Potassium", "Phosphorous")
    @classmethod
    def must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v


class FertilizerResponse(BaseModel):
    predicted_fertilizer: str
    confidence:           float


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {
        "message": "Smart Fertilizer Recommendation API is running.",
        "docs":    "/docs",
    }


@app.get("/categories", tags=["General"])
def get_categories():
    """Return all valid string values for Soil Type and Crop Type."""
    if le_soil is None or le_crop is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {
        "soil_types": list(le_soil.classes_),
        "crop_types": list(le_crop.classes_),
    }


@app.post("/predict", response_model=FertilizerResponse, tags=["Prediction"])
def predict(request: FertilizerRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Encode categorical string inputs → integers
    try:
        soil_encoded = int(le_soil.transform([request.Soil_Type])[0])
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Soil Type: '{request.Soil_Type}'. "
                   f"Valid options: {list(le_soil.classes_)}"
        )

    try:
        crop_encoded = int(le_crop.transform([request.Crop_Type])[0])
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown Crop Type: '{request.Crop_Type}'. "
                   f"Valid options: {list(le_crop.classes_)}"
        )

    # Build input dict with encoded values
    raw = {
        "Temparature": request.Temparature,
        "Humidity":    request.Humidity,
        "Moisture":    request.Moisture,
        "Soil Type":   soil_encoded,
        "Crop Type":   crop_encoded,
        "Nitrogen":    request.Nitrogen,
        "Potassium":   request.Potassium,
        "Phosphorous": request.Phosphorous,
    }

    # Reorder to match training column order
    try:
        input_df = pd.DataFrame([raw])[column_names]
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature mismatch: {e}. Expected columns: {column_names}"
        )

    # Predict
    encoded_pred = pipeline.predict(input_df)[0]
    proba        = pipeline.predict_proba(input_df)[0]

    return FertilizerResponse(
        predicted_fertilizer=str(le_target.inverse_transform([encoded_pred])[0]),
        confidence=round(float(proba[encoded_pred]), 4),
    )
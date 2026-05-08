print("STARTED")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import os
import shap
from google import genai # Updated import
from dotenv import load_dotenv


load_dotenv(override=True)
# Initialize the new GenAI Client
# It automatically looks for the GEMINI_API_KEY environment variable if you don't pass it explicitly.
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "Model")

rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

# Initialize the SHAP explainer globally
explainer = shap.TreeExplainer(rf_model)

app = FastAPI()

class CropInput(BaseModel):
    N: float = Field(..., ge=0, description="Nitrogen content in soil")
    P: float = Field(..., ge=0, description="Phosphorus content in soil")
    K: float = Field(..., ge=0, description="Potassium content in soil")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity in %")
    ph: float = Field(..., ge=0, le=14, description="pH value of soil")
    rainfall: float = Field(..., ge=0, description="Rainfall in mm")

class CropExplainOutput(BaseModel):
    recommended_crop: str
    explanation: str

def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    df['NPK_ratio'] = df['N'] / (df['P'] + df['K'] + 1)
    df['total_nutrients'] = df['N'] + df['P'] + df['K']
    df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
    df['rainfall_humidity_ratio'] = df['rainfall'] / (df['humidity'] + 1)
    return df[feature_names]

@app.get("/")
def root():
    return {"message": "Crop Recommendation API is running. Visit /docs for the interactive UI."}

@app.post("/predict", response_model=CropExplainOutput)
def predict_and_explain(input_data: CropInput):
    try:
        raw_inputs = input_data.model_dump()
        df = engineer_features(raw_inputs)
        
        # 1. Make Prediction
        prediction_idx = rf_model.predict(df)[0]
        crop = label_encoder.inverse_transform([prediction_idx])[0]
        
        # 2. Calculate SHAP Values
        shap_values = explainer.shap_values(df)
        
        if isinstance(shap_values, list):
            class_shap_values = shap_values[prediction_idx][0]
        else:
            class_shap_values = shap_values[0, :, prediction_idx]
            
        feature_impacts = list(zip(feature_names, class_shap_values))
        feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        top_4 = feature_impacts[:4]
        
        shap_summary = ", ".join([f"{feat} ({'+' if val > 0 else '-'}{abs(val):.2f})" for feat, val in top_4])
        input_summary = ", ".join([f"{k}={v}" for k, v in raw_inputs.items()])

        # 3. Generate Physiological Explanation with the new Gemini SDK
        system_prompt = f"""
        You are an expert agronomist explaining a machine learning crop recommendation to a fellow agriculture engineer.
        The model recommended '{crop.upper()}' based on these exact conditions: {input_summary}.
        
        The top 4 mathematical drivers (SHAP values) for this decision are: {shap_summary}.
        (Positive values pushed the model toward this crop; negative values pushed it away).
        
        Write a concise, insightful explanation (under 150 words) focusing strictly on plant biology and physiology. Explain WHY these specific environmental or nutrient values physically make sense for {crop.upper()}'s biological needs (e.g., nodule formation, root development, transpiration, drought tolerance). 
        Do not just repeat the numbers; explain the biological 'why' behind them.
        """
        
        # New API Call Syntax
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=system_prompt,
        )
        
        return CropExplainOutput(
            recommended_crop=crop,
            explanation=response.text.strip()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
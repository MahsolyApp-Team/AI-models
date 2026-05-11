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

# Fallback chain: if one hits rate limit, try the next
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

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
        You are a friendly agricultural advisor explaining a crop recommendation to a farmer with no technical background.
        The model recommended '{crop.upper()}' based on these conditions: {input_summary}.
        
        The main factors that led to this recommendation are: {shap_summary}.
        
        Write a simple, friendly explanation (under 120 words) that answers: "Why is {crop.upper()} the right choice for these conditions?"
        
        Rules:
        - Use plain everyday language. No jargon, no scientific terms.
        - Focus on practical reasons a farmer would understand (e.g., "this crop handles dry weather well" instead of "exhibits drought tolerance via stomatal regulation").
        - If you must use a technical term, explain it in one simple phrase right after.
        - Do NOT mention SHAP values, model, or numbers — just explain the 'why' in plain words.
        - Keep a warm, helpful tone as if talking to a friend.
        """
        
        # Try each Gemini model in order until one succeeds
        explanation = None
        for model in GEMINI_MODELS:
            try:
                print(f"Trying {model}...")
                response = client.models.generate_content(
                    model=model,
                    contents=system_prompt,
                )
                explanation = response.text.strip()
                break
            except Exception:
                continue  # rate limit or error → try next model

        return CropExplainOutput(
            recommended_crop=crop,
            explanation=explanation or "Sorry, explanation is not available right now."
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
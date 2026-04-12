from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('lightgbm_crop_model.joblib')

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.get("/")
def read_root():
    return {"message": "Mahsoly Crop Recommendation API"}

@app.post("/predict")
def predict_crop(data: CropInput):
    input_df = pd.DataFrame([data.dict()])
    
    prediction = model.predict(input_df)
    
    return {"recommended_crop": str(prediction[0])}
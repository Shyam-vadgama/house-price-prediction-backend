from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

# Replace with your Hugging Face Space URL
HF_API_URL = "https://s-h-y-a-m-123-housing-price-predictor.hf.space/run/predict"

class HouseFeatures(BaseModel):
    rooms: int
    area: float
    location: str

@app.get("/")
def root():
    return {"message": "Housing Price Predictor API running!"}

@app.post("/predict")
def predict(features: HouseFeatures):
    # Hugging Face payload format
    payload = {
        "data": [
            features.rooms,
            features.area,
            features.location
        ]
    }

    try:
        response = requests.post(HF_API_URL, json=payload)
        result = response.json()
        return {"prediction": result.get("data", [])[0]}
    except Exception as e:
        return {"error": str(e)}

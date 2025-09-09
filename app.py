from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Using wildcard for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace with your Hugging Face Space URL
HF_API_URL = "https://huggingface.co/spaces/s-h-y-a-m-123/housing-price-predictor"

# 1. Pydantic Model ko 12 parameters ke saath update kiya
class HouseFeatures(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str  # "Yes" or "No"
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str # "Furnished", "Semi-furnished", "Unfurnished"

@app.get("/")
def root():
    return {"message": "Housing Price Predictor API running!"}

@app.post("/predict")
def predict(features: HouseFeatures):
    
    # 2. Payload ko 12 parameters ke saath sahi order mein banaya
    payload = {
        "data": [
            features.area,
            features.bedrooms,
            features.bathrooms,
            features.stories,
            features.mainroad,
            features.guestroom,
            features.basement,
            features.hotwaterheating,
            features.airconditioning,
            features.parking,
            features.prefarea,
            features.furnishingstatus
        ]
    }

    try:
        # API endpoint /predict_price hai
        response = requests.post(f"{HF_API_URL}/predict_price", json=payload)
        response.raise_for_status()
        result = response.json()
        
        prediction_data = result.get("data")
        if prediction_data and len(prediction_data) > 0:
            return {"prediction": prediction_data[0]}
        else:
            return {"error": f"No data received from HF. Response: {result}"}
            
    except Exception as e:
        return {"error": str(e)}

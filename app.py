from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware # <-- Ye import karo

app = FastAPI()

# CORS Middleware Settings
# Yahan apne frontend ka URL daalna
origins = [
    "http://localhost",
    "http://localhost:3000", # Agar local pe test kar rahe ho
    "http://localhost:5173", # Vite/React default local server
    "https://house-price-prediction-frontend.netlify.app/" # <-- YAHAN APNA NETLIFY URL DAALO
]

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],  # Saare methods (GET, POST, etc.) allow karega
    allow_headers=["*"],  # Saare headers allow karega
)


# Replace with your Hugging Face Space URL
HF_API_URL = "https://huggingface.co/spaces/s-h-y-a-m-123/housing-price-predictor"

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
        response = requests.post(f"{HF_API_URL}/predict_price", json=payload)
        response.raise_for_status()
        result = response.json()
        
        # --- BEHTAR CHECK YAHAN ADD KIYA HAI ---
        prediction_data = result.get("data")
        
        if prediction_data and len(prediction_data) > 0:
            # Agar data list mein kuch hai, to pehla item bhejo
            return {"prediction": prediction_data[0]}
        else:
            # Agar HF se data nahi aaya ya khaali list aayi
            return {"error": f"No data received from Hugging Face. API Response: {result}"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Error calling Hugging Face API: {e}"}
    except Exception as e:
        return {"error": str(e)}

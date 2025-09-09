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
        # Note: Hugging Face Spaces API endpoint usually needs "/run/predict"
        # Double check this from your HF Space's API page.
        response = requests.post(f"{HF_API_URL}/run/predict", json=payload)
        response.raise_for_status() # Ye line add karna accha hai, error check ke liye
        result = response.json()
        return {"prediction": result.get("data", [])[0]}
    except requests.exceptions.RequestException as e:
        # Better error handling for network issues
        return {"error": f"Error calling Hugging Face API: {e}"}
    except Exception as e:
        return {"error": str(e)}

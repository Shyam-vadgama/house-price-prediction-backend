from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, exceptions  # <-- `Client` ko import karo

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Using wildcard for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Model ko 12 parameters ke saath update kiya
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
    try:
        # 1. Gradio client initialize karo
        #    Yeh Space ID "s-h-y-a-m-123/housing-price-predictor" se connect karega
        client = Client("s-h-y-a-m-123/housing-price-predictor")
        
        # 2. Predict function call karo, bilkul simple tarike se
        #    Yeh saara queue aur session ka kaam automatically handle karega
        result = client.predict(
            area=features.area,
            bedrooms=features.bedrooms,
            bathrooms=features.bathrooms,
            stories=features.stories,
            mainroad=features.mainroad,
            guestroom=features.guestroom,
            basement=features.basement,
            hotwaterheating=features.hotwaterheating,
            airconditioning=features.airconditioning,
            parking=features.parking,
            prefarea=features.prefarea,
            furnishingstatus=features.furnishingstatus,
            api_name="/predict_price"  # API ka naam batana zaroori hai
        )
        
        # 3. Prediction result bhejo
        return {"prediction": result}

    except exceptions.AppError as e:
        # Gradio se aane wale specific errors ko handle karo
        return {"error": f"Gradio App Error: {e}"}
    except Exception as e:
        # Baaki saare errors (jaise network issue) yahan handle honge
        return {"error": str(e)}

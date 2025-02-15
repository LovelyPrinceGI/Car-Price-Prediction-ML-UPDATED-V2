from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Initialize the FastAPI app
app = FastAPI(title="Car Price Prediction UI", description="UI for predicting car prices", version="1.0")

# Setup middleware
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the saved model and preprocessors
MODEL_PATH = os.path.join("model", "car_price_prediction.model")
BRAND_ENCODER_PATH = os.path.join("preprocess", "brandre_encoder.prep")
SCALER_PATH = os.path.join("preprocess", "scaler.prep")

model = pickle.load(open(MODEL_PATH, "rb"))
brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# Define the input data structure
class CarPriceInput(BaseModel):
    max_power: float
    fuel_eff: float
    year: int
    brand_region: str

# Home route
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "predicted_price": None})

# Prediction route

# Update the predict route to use Form data instead of JSON
@app.post("/predict/")
def predict_car_price(
    request: Request,
    max_power: float = Form(...),
    year: int = Form(...),
    km_driven: float = Form(...),
    fuel_eff: float = Form(...),
    brand_region: str = Form(...)
):
    
    try:
        vehicle_specs = {
            'Max Power': max_power,
            'Year': year,
            'Km Driven': km_driven,
            'Fuel Efficiency': fuel_eff,
            'Brand Region': brand_region
        }
        # Transform the input data
        brand_region_encoded = brand_encoder.transform([[brand_region]])
        brand_region_array = brand_region_encoded[0]

        # Prepare input for the model
        features = np.array([[max_power, year, km_driven, fuel_eff] + list(brand_region_array)])
        print(brand_region_array)
        features_scaled = scaler.transform(features[:, :2])  # Scale numerical features (fuel_eff and year)

        # Merge scaled features with categorical features
        final_features = np.hstack([features_scaled, features[:, 2:]])

        # Predict price
        predicted_price = model.predict(final_features)[0]
        predicted_price = np.exp(predicted_price)  # Undo log transformation if applied during training

        return templates.TemplateResponse("index.html", {"request": request, "predicted_price": predicted_price, 'vehicle_specs': vehicle_specs})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")

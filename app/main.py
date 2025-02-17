from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import sys
import os
import uvicorn
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'LinearRegression':
            from regression import LinearRegression
            return LinearRegression
        return super().find_class(module, name)

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

# Load the old and new models
OLD_MODEL_PATH = os.path.join("model", "car_price_prediction.model")
NEW_MODEL_PATH = os.path.join("new_model_v2", "Best_GridSearch_Model.pkl")

# Load preprocessors
BRAND_ENCODER_PATH = os.path.join("preprocess_v2", "brandre_encoder_v2.prep")
SCALER_PATH = os.path.join("preprocess_v2", "scaler_v2.prep")

old_model = pickle.load(open(OLD_MODEL_PATH, "rb"))
new_model = CustomUnpickler(open(NEW_MODEL_PATH, "rb")).load() 
brand_encoder = pickle.load(open(BRAND_ENCODER_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

# Define the input data structure
class CarPriceInput(BaseModel):
    max_power: float
    fuel_eff: float
    year: int
    km_driven: float
    brand_region: str

# Home route
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "predicted_price": None, "vehicle_specs": None})

# Prediction route with model selection
@app.post("/predict/old")
def predict_car_price_old(
    request: Request,
    max_power: float = Form(...),
    year: int = Form(...),
    km_driven: float = Form(...),
    fuel_eff: float = Form(...),
    brand_region: str = Form(...)
):
    try:
        model = pickle.load(open("model/car_price_prediction.model", "rb"))
        brand_encoder = pickle.load(open("preprocess_v2/brandre_encoder_v2.prep", "rb"))
        scaler = pickle.load(open("preprocess_v2/scaler_v2.prep", "rb"))

        # Process input and predict
        brand_region_encoded = brand_encoder.transform([[brand_region]])
        brand_region_array = brand_region_encoded[0]

        features = np.array([[max_power, year, km_driven, fuel_eff] + list(brand_region_array)])
        features[0][0] = np.log1p(features[0][0])  # log(1 + x) to handle 0 values
        features[0][2] = np.log1p(features[0][2])
        features_scaled = scaler.transform(features[:, :2])
        final_features = np.hstack([features_scaled, features[:, 2:]])

        predicted_price = model.predict(final_features)[0]
        predicted_price = np.exp(predicted_price)

        vehicle_specs = {
            "Max Power": max_power,
            "Year": year,
            "Km Driven": km_driven,
            "Fuel Efficiency": fuel_eff,
            "Brand Region": brand_region
        }

        return templates.TemplateResponse("index.html", {"request": request, "predicted_price": predicted_price, "vehicle_specs": vehicle_specs})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")


@app.post("/predict/new")
def predict_car_price_new(
    request: Request,
    max_power: float = Form(...),
    year: int = Form(...),
    km_driven: float = Form(...),
    fuel_eff: float = Form(...),
    brand_region: str = Form(...)
):
    try:
        model = CustomUnpickler(open("new_model_v2/Best_GridSearch_Model.pkl", "rb")).load()
        brand_encoder = pickle.load(open("preprocess_v2/brandre_encoder_v2.prep", "rb"))
        scaler = pickle.load(open("preprocess_v2/scaler_v2.prep", "rb"))

        # Process input and predict
        brand_region_encoded = brand_encoder.transform([[brand_region]])
        brand_region_array = brand_region_encoded[0]

        features = np.array([[max_power, year, km_driven, fuel_eff] + list(brand_region_array)])
        # Log transform 'max_power', 'km_drivn'
        features[0][0] = np.log1p(features[0][0])  # log(1 + x) to handle 0 values
        features[0][2] = np.log1p(features[0][2])
        features_scaled = scaler.transform(features[:, :2])
        final_features = np.hstack([features_scaled, features[:, 2:]])

        predicted_price = model.predict(final_features)[0]
        predicted_price = np.exp(predicted_price)

        vehicle_specs = {
            "Max Power": max_power,
            "Year": year,
            "Km Driven": km_driven,
            "Fuel Efficiency": fuel_eff,
            "Brand Region": brand_region
        }

        return templates.TemplateResponse("index.html", {"request": request, "predicted_price": predicted_price, "vehicle_specs": vehicle_specs})

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")

if __name__=="__main__":
    uvicorn.run(
    "main:app",
    host="0.0.0.0",
    port=8000,
    reload=True
)
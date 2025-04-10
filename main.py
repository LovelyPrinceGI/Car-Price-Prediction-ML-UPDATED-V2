from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import sys
import os
import uvicorn
import pickle



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

# Load the model from MLflow
mlflow.set_tracking_uri("https://admin:password@mlflow.ml.brain.cs.ait.ac.th")
model_uri = "models:/st124876-a3-model/4"
model = mlflow.pyfunc.load_model(model_uri)

# Load preprocessors
BRAND_ENCODER_PATH = os.path.join("preprocess_v2", "brandre_encoder_v2.prep")
SCALER_PATH = os.path.join("preprocess_v2", "scaler_v2_updated.prep")


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
    return templates.TemplateResponse("index.html", {"request": request, "predicted_category": None, "vehicle_specs": None})

# Prediction route with model selection
@app.post("/predict/custom_logistic_regression")
def predict_car_price_old(
    request: Request,
    max_power: float = Form(...),
    year: int = Form(...),
    km_driven: float = Form(...),
    fuel_eff: float = Form(...),
    brand_region: str = Form(...)
):
    try:

        brand_encoder = pickle.load(open("preprocess_v2/brandre_encoder_v2.prep", "rb"))
        scaler_v2_updated = pickle.load(open("preprocess_v2/scaler_v2_updated.prep", "rb"))

        # Process input and predict
        brand_region_encoded = brand_encoder.transform([[brand_region]])
        brand_region_array = brand_region_encoded[0]

        # Combine all inputs
        numeric_features = np.array([[max_power, year, km_driven, fuel_eff]])  # shape (1, 4)

        # Apply Log transform
        numeric_features[0][0] = np.log1p(numeric_features[0][0])  # max_power
        numeric_features[0][2] = np.log1p(numeric_features[0][2])  # km_driven

        # Power transform fuel_eff
        fuel_eff_transformer = pickle.load(open("preprocess_v2/fuel_eff_transformer.prep", "rb"))
        numeric_features[0][3] = fuel_eff_transformer.transform([[numeric_features[0][3]]])[0][0]

        # Normalize using scaler
        scaled_numeric_features = scaler_v2_updated.transform(numeric_features)[0]  # shape: (4,)

        # Concatenate one-hot encoded brand_region
        final_features = np.hstack([scaled_numeric_features, brand_region_array]).reshape(1,-1)
        print(final_features)


        # Make a prediction
        predicted_category = model.predict(final_features)
        # predicted_category = np.exp(predicted_category)

        vehicle_specs = {
            "Max Power": max_power,
            "Year": year,
            "Km Driven": km_driven,
            "Fuel Efficiency": fuel_eff,
            "Brand Region": brand_region
        }

        category_range_map = {
            0: "Below ฿260,000",
            1: "฿260,000 - ฿450,000",
            2: "฿450,000 - ฿680,000",
            3: "More than ฿680,000"
        }

        return templates.TemplateResponse("index.html", {
            "request": request,
            "predicted_category": predicted_category,
            "category_range": category_range_map.get(predicted_category, "Unknown"),
            "vehicle_specs": {
                "max_power": max_power,
                "year": year,
                "km_driven": km_driven,
                "fuel_eff": fuel_eff,
                "brand_region": brand_region
            }
        })


    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
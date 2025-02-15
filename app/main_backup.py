from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Initialize the FastAPI app
app = FastAPI(title="Car Price Prediction API", description="API for predicting car prices", version="1.0")

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
@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API!"}

# Prediction route
@app.post("/predict/")
def predict_car_price(input_data: CarPriceInput):
    # Validate and preprocess the input
    try:
        # Transform the input data
        brand_region_encoded = brand_encoder.transform([[input_data.brand_region]])
        brand_region_array = brand_region_encoded[0]

        # Prepare input for the model
        features = np.array([[input_data.max_power, input_data.year, input_data.fuel_eff] + list(brand_region_array)])
        features_scaled = scaler.transform(features[:, :2])  # Scale numerical features (fuel_eff and year)

        # Merge scaled features with categorical features
        final_features = np.hstack([features_scaled, features[:, 2:]])

        # Predict price
        predicted_price = model.predict(final_features)[0]
        predicted_price = np.exp(predicted_price)  # Undo log transformation if applied during training

        return {"predicted_price": predicted_price}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {e}")
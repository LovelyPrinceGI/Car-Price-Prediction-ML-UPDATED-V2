import pickle
import numpy as np

# Load model and preprocessors
model = pickle.load(open("model/car_price_prediction.model", "rb"))
brand_encoder = pickle.load(open("preprocess/brandre_encoder.prep", "rb"))
scaler = pickle.load(open("preprocess/scaler.prep", "rb"))

# Inputs
max_power = 102.0
year = 2013
km_driven = 10000.0
fuel_eff = 21.0
brand_region = "Asia"

try:
    # Transform the input data
    brand_region_encoded = brand_encoder.transform([[brand_region]])
    brand_region_array = brand_region_encoded[0]

    # Prepare input for the model
    features = np.array([[max_power, year, km_driven, fuel_eff] + list(brand_region_array)])
    print("Features Before Scaling:", features)

    # Scale numerical features (year and fuel_eff)
    features[:, [1, 3]] = scaler.transform(features[:, [1, 3]])
    print("Features After Scaling:", features)

    # Predict price
    predicted_price = model.predict(features)[0]
    predicted_price = np.exp(predicted_price)  # Undo log transformation if applied during training

    print(f"Predicted Price: {predicted_price}")
    print(f"Vehicle Specs: Max Power: {max_power}, Year: {year}, Km Driven: {km_driven}, "
          f"Fuel Efficiency: {fuel_eff}, Brand Region: {brand_region}")

except Exception as e:
    print(f"Error during prediction: {e}")



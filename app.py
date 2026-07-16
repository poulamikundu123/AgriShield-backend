from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from pydantic import BaseModel
from PIL import Image

import joblib
import numpy as np
import json

# ==========================================================
# CREATE FASTAPI APP
# ==========================================================

app = FastAPI(
    title="Multimodal Crop Yield & Disease Prediction API"
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# LOAD AI MODELS
# ==========================================================

print("=" * 60)

print("Loading Disease Model...")
disease_model = load_model("../models/plant_disease_model.h5")
print("✅ Disease Model Loaded Successfully!")

print("Loading Yield Model...")
yield_model = joblib.load("../models/yield_model.pkl")
print("✅ Yield Model Loaded Successfully!")

print("Loading Crop Encoder...")
encoder = joblib.load("../models/crop_encoder.pkl")
print("✅ Crop Encoder Loaded Successfully!")

# ==========================================================
# LOAD JSON FILES
# ==========================================================

print("Loading Class Names...")

with open("../data/class_names.json", "r") as f:
    class_names = json.load(f)

print("✅ Class Names Loaded!")

print("Loading Loss Table...")

with open("../data/loss_table.json", "r") as f:
    loss_table = json.load(f)

print("✅ Loss Table Loaded!")

print("Loading Pesticide Table...")

with open("../data/pesticide.json", "r") as f:
    pesticide_table = json.load(f)

print("✅ Pesticide Table Loaded!")

print("=" * 60)

# ==========================================================
# PYDANTIC MODEL
# ==========================================================

class YieldInput(BaseModel):
    crop: str
    nitrogen: float
    phosphorus: float
    potassium: float
    soil_ph: float
    temperature: float
    humidity: float
    rainfall: float
    land_area: float

# ==========================================================
# HELPER FUNCTION - DISEASE PREDICTION
# ==========================================================

def predict_disease_from_image(image):

    # Resize image
    image = image.resize((224, 224))

    # Convert to NumPy
    image = np.array(image)

    # Normalize
    image = image / 255.0

    # Add Batch Dimension
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = disease_model.predict(image, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction) * 100)

    disease = class_names[predicted_index]

    return disease, confidence

# ==========================================================
# HELPER FUNCTION - YIELD PREDICTION
# ==========================================================

def predict_crop_yield(
    crop,
    nitrogen,
    phosphorus,
    potassium,
    soil_ph,
    temperature,
    humidity,
    rainfall,
    land_area
):

    crop_encoded = encoder.transform([crop])[0]

    input_data = [[
        crop_encoded,
        nitrogen,
        phosphorus,
        potassium,
        soil_ph,
        temperature,
        humidity,
        rainfall,
        land_area
    ]]

    prediction = yield_model.predict(input_data)

    return float(prediction[0])

# ==========================================================
# HOME API
# ==========================================================

@app.get("/")
def home():
    return {
        "message": "Multimodal Crop AI Backend Running Successfully!"
    }

# ==========================================================
# DISEASE PREDICTION API
# ==========================================================

@app.post("/predict-disease")
async def predict_disease(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")

    disease, confidence = predict_disease_from_image(image)

    return {
        "Disease": disease,
        "Confidence": round(confidence, 2)
    }

# ==========================================================
# YIELD PREDICTION API
# ==========================================================

@app.post("/predict-yield")
def predict_yield(data: YieldInput):

    predicted_yield = predict_crop_yield(
        data.crop,
        data.nitrogen,
        data.phosphorus,
        data.potassium,
        data.soil_ph,
        data.temperature,
        data.humidity,
        data.rainfall,
        data.land_area
    )

    return {
        "Crop": data.crop,
        "Expected Yield (ton/hectare)": round(predicted_yield, 2)
    }

# ==========================================================
# FINAL MULTIMODAL PREDICTION API
# ==========================================================

@app.post("/predict-final")
async def predict_final(

    file: UploadFile = File(...),

    crop: str = Form(...),
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    soil_ph: float = Form(...),
    temperature: float = Form(...),
    humidity: float = Form(...),
    rainfall: float = Form(...),
    land_area: float = Form(...)

):

    # -----------------------------
    # Read Image
    # -----------------------------

    image = Image.open(file.file).convert("RGB")

    # -----------------------------
    # Disease Prediction
    # -----------------------------

    disease, confidence = predict_disease_from_image(image)

    # -----------------------------
    # Yield Prediction
    # -----------------------------

    expected_yield = predict_crop_yield(

        crop,
        nitrogen,
        phosphorus,
        potassium,
        soil_ph,
        temperature,
        humidity,
        rainfall,
        land_area

    )

    # -----------------------------
    # Get Disease Information
    # -----------------------------

    disease_info = loss_table.get(

        disease,

        {
            "loss": 0,
            "severity": "Unknown"
        }

    )

    loss_percent = disease_info["loss"]

    severity = disease_info["severity"]

    # -----------------------------
    # Calculate Final Yield
    # -----------------------------

    final_yield = expected_yield * (1 - loss_percent / 100)

    # -----------------------------
    # Get Pesticide Recommendation
    # -----------------------------

    recommendation = pesticide_table.get(

        disease,

        {
            "Pesticides": ["No Recommendation"],
            "Advice": ["Consult an agricultural expert."]
        }

    )

    # -----------------------------
    # Return Final Result
    # -----------------------------

    return {

        "Crop": crop,

        "Disease": disease,

        "Confidence": round(confidence, 2),

        "Severity": severity,

        "Expected Yield (ton/hectare)": round(expected_yield, 2),

        "Yield Loss (%)": loss_percent,

        "Final Yield (ton/hectare)": round(final_yield, 2),

        "Recommended Pesticides": recommendation["Pesticides"],

        "Advice": recommendation["Advice"],

        "Temperature": temperature,

        "Humidity": humidity,

        "Rainfall": rainfall

    }
from fastapi import FastAPI
import pandas as pd
import joblib
import logging

from api.schema import HeartDiseaseRequest

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Heart Disease Prediction API"
)

# Load artifacts
model = joblib.load("models/final_model.pkl")
scaler = joblib.load("models/scaler.pkl")

with open("models/champion.txt", "r") as f:
    champion_name = f.read()


@app.get("/")
def home():

    logging.info("Health check endpoint called")

    return {
        "message": "Heart Disease Prediction API",
        "champion_model": champion_name
    }


@app.post("/predict")
def predict(data: HeartDiseaseRequest):

    logging.info("Prediction request received")

    input_df = pd.DataFrame([
        data.model_dump()
    ])

    if "LogReg" in champion_name:
        input_df = scaler.transform(input_df)

    prediction = model.predict(input_df)[0]

    probability = max(
        model.predict_proba(input_df)[0]
    )

    logging.info(
        f"Prediction={prediction}, Confidence={probability:.4f}")

    return {
        "prediction": int(prediction),
        "confidence": float(probability)
    }
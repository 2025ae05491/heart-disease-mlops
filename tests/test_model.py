import joblib
import pandas as pd

def test_model_loads():

    model = joblib.load(
        "models/final_model.pkl"
    )

    assert model is not None

def test_scaler_loads():

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    assert scaler is not None

def test_prediction_runs():

    model = joblib.load(
        "models/final_model.pkl"
    )

    df = pd.read_csv(
        "data/processed/heart_clean.csv"
    )

    sample = df.drop(
        "target",
        axis=1
    ).iloc[[0]]

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib

from feature_engineering import engineer_features

app = FastAPI(
    title="Inventory Supply–Demand Risk API",
    description="Predict high-risk inventory items using ML",
    version="1.0"
)

# Load model artifacts
model = joblib.load("models/mismatch_risk_model.pkl")
scaler = joblib.load("models/scaler.pkl")

FEATURE_COLS = [
    "reorder_level",
    "reorder_quantity",
    "unit_price",
    "reorder_pressure",
    "inventory_age_days",
    "days_since_last_order"
]

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict_risk(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    df = engineer_features(df)

    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[:, 1]

    df["high_risk_probability"] = probs
    df["risk_label"] = df["high_risk_probability"].apply(
        lambda x: "High" if x >= 0.7 else "Medium" if x >= 0.4 else "Low"
    )

    return df.to_dict(orient="records")

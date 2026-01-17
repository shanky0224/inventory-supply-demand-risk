from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os

from feature_engineering import engineer_features

app = FastAPI(
    title="Inventory Supply–Demand Risk API",
    description="Predict high-risk inventory items using ML",
    version="1.0"
)

# ✅ Robust path handling (works locally + Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /src/api
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # /src
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

model = joblib.load(os.path.join(MODEL_DIR, "mismatch_risk_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

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

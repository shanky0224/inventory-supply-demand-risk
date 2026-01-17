from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os

from feature_engineering import engineer_features

# --------------------------------------------------
# App config
# --------------------------------------------------
app = FastAPI(
    title="Inventory Supply–Demand Risk API",
    description="Predict high-risk inventory items using ML",
    version="1.0"
)

# --------------------------------------------------
# Robust path handling (local + Render)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # /project/api
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "mismatch_risk_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# --------------------------------------------------
# Load model artifacts
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# Feature columns (MUST match training)
# --------------------------------------------------
FEATURE_COLS = [
    "reorder_level",
    "reorder_quantity",
    "unit_price",
    "reorder_pressure",
    "inventory_age_days",
    "days_since_last_order"
]

# --------------------------------------------------
# Health check
# --------------------------------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}

# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------
@app.post("/predict")
def predict_risk(file: UploadFile = File(...)):
    # Read uploaded CSV
    df = pd.read_csv(file.file)

    # Feature engineering
    df = engineer_features(df)

    # Prepare model input
    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    # Predict probabilities
    probs = model.predict_proba(X_scaled)[:, 1]
    df["high_risk_probability"] = probs

    # Threshold logic
    threshold = 0.5
    high_risk_df = df[df["high_risk_probability"] >= threshold]

    # API response
    response = {
        "total_products": int(len(df)),
        "high_risk_items": int(len(high_risk_df)),
        "high_risk_percentage": round(
            (len(high_risk_df) / len(df)) * 100, 2
        ),
        "top_high_risk_items": (
            high_risk_df
            .sort_values("high_risk_probability", ascending=False)
            .head(20)
            .to_dict(orient="records")
        )
    }

    return response

from fastapi import FastAPI, UploadFile, File, Query
import pandas as pd
import joblib
import os

from feature_engineering import engineer_features

app = FastAPI(
    title="Inventory Supply–Demand Risk API",
    description="Predict high-risk inventory items using ML",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

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
def predict_risk(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    df = pd.read_csv(file.file)

    df = engineer_features(df)

    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[:, 1]
    df["high_risk_probability"] = probs

    high_risk_df = df[df["high_risk_probability"] >= threshold]

    return {
        "total_products": int(len(df)),
        "high_risk_items": int(len(high_risk_df)),
        "high_risk_percentage": round(len(high_risk_df) / len(df) * 100, 2),
        "top_high_risk_items": (
            high_risk_df
            .sort_values("high_risk_probability", ascending=False)
            .head(20)
            .to_dict(orient="records")
        )
    }

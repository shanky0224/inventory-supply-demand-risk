import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Inventory Risk Prediction",
    page_icon="📦",
    layout="wide"
)

# -----------------------------
# Load model artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/mismatch_risk_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# -----------------------------
# Model feature columns
# -----------------------------
FEATURE_COLS = [
    "reorder_level",
    "reorder_quantity",
    "unit_price",
    "reorder_pressure",
    "inventory_age_days",
    "days_since_last_order"
]

# -----------------------------
# Feature Engineering
# -----------------------------
def engineer_features(df):
    df = df.copy()
    df.columns = df.columns.str.lower()

    # Clean unit price ($ → float)
    df["unit_price"] = (
        df["unit_price"]
        .astype(str)
        .str.replace("$", "", regex=False)
    )
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce").fillna(0)

    # Dates
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["last_order_date"] = pd.to_datetime(df["last_order_date"], errors="coerce")

    today = pd.Timestamp.today()

    df["inventory_age_days"] = (today - df["date_received"]).dt.days.fillna(0)
    df["days_since_last_order"] = (today - df["last_order_date"]).dt.days.fillna(0)

    df["reorder_pressure"] = df["reorder_level"] - df["stock_quantity"]

    return df

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")
threshold = st.sidebar.slider(
    "High-Risk Threshold",
    min_value=0.2,
    max_value=0.8,
    value=0.5,
    step=0.05
)

st.sidebar.markdown("""
**Threshold Logic**
- Lower → catch more risky items
- Higher → fewer false alarms
""")

# -----------------------------
# Main UI
# -----------------------------
st.title("📦 Inventory Supply–Demand Risk Prediction")
st.markdown("Predict **high-risk inventory items** to enable proactive stock decisions.")

uploaded_file = st.file_uploader("Upload Inventory CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df_raw.head())

    # Feature engineering
    df = engineer_features(df_raw)

    # Validate columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # -----------------------------
    # ✅ CORRECT BINARY PREDICTION
    # -----------------------------
    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)

    # Probability of HIGH RISK (class 1)
    df["high_risk_probability"] = probs[:, 1]

    df["high_risk_flag"] = (df["high_risk_probability"] >= threshold).astype(int)

    df["risk_label"] = pd.qcut(
        df["high_risk_probability"],
        q=[0, 0.6, 0.85, 1.0],
        labels=["Low", "Medium", "High"]
    )
    
    df["high_risk_flag"] = (df["risk_label"] == "High").astype(int)



    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("🚨 High-Risk Inventory Items")

    high_risk_df = df[df["high_risk_flag"] == 1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Products", len(df))
    c2.metric("High-Risk Items", len(high_risk_df))
    c3.metric("High-Risk %", f"{len(high_risk_df)/len(df)*100:.1f}%")

    st.dataframe(
        high_risk_df.sort_values("high_risk_probability", ascending=False)
    )

    st.download_button(
        "⬇️ Download Predictions",
        data=df.to_csv(index=False),
        file_name="inventory_risk_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("👆 Upload a CSV file to begin prediction")

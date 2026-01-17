import streamlit as st
import requests

API_URL = "https://inventory-risk-api.onrender.com/predict"

st.set_page_config(page_title="Inventory Risk Dashboard", layout="wide")

st.title("📦 Inventory Supply–Demand Risk Dashboard")
st.markdown("Streamlit frontend connected to FastAPI backend")

uploaded_file = st.file_uploader("Upload Inventory CSV", type=["csv"])
threshold = st.slider("High-Risk Threshold", 0.2, 0.8, 0.5, 0.05)

if uploaded_file:
    with st.spinner("Sending data to FastAPI..."):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "text/csv"
            )
        }

        response = requests.post(
            API_URL,
            files=files,
            params={"threshold": threshold}
        )

    if response.status_code == 200:
        result = response.json()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Products", result["total_products"])
        c2.metric("High-Risk Items", result["high_risk_items"])
        c3.metric("High-Risk %", f'{result["high_risk_percentage"]}%')

        st.subheader("🚨 Top High-Risk Items")
        st.dataframe(result["top_high_risk_items"])
    else:
        st.error("API Error")

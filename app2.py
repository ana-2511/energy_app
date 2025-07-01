import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import plot_importance
import requests

# Set up the app layout
st.set_page_config(page_title="Smart Home Energy Dashboard", layout="centered")
st.title("🏠 Smart Home Appliance Energy Predictor")
st.markdown("Use this AI-powered tool to forecast your energy usage and explore trends from your smart home data.")

# --- Tabs ---
tabs = st.tabs(["📊 Predict Usage", "📈 Upload CSV Trend", "🔍 Feature Importance"])

# --- Tab 1: Predict Usage ---
with tabs[0]:
    st.header("🔧 Forecast with Live Inputs")

    col1, col2 = st.columns(2)
    with col1:
        hour = st.slider("🕒 Hour of Day", 0, 23, 14)
        day_of_week = st.selectbox("📅 Day of the Week", 
            [0,1,2,3,4,5,6], format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])
        is_weekend = 1 if day_of_week in [5, 6] else 0
        user_activity = st.radio("🚶 Activity Level", ["Low", "Normal", "High"])

    with col2:
        T_out = st.slider("🌡️ Outdoor Temp (°C)", -10.0, 40.0, 22.0)
        RH_out = st.slider("💧 Humidity (%)", 10.0, 100.0, 55.0)
        Visibility = st.slider("👀 Visibility (km)", 0.0, 66.0, 40.0)
        Tdewpoint = st.slider("💨 Dew Point (°C)", -10.0, 30.0, 10.0)

    # Map user-friendly input to internal engineered features
    if user_activity == "Low":
        lag1, lag24, roll3, roll6 = 150, 140, 145, 143
    elif user_activity == "High":
        lag1, lag24, roll3, roll6 = 400, 380, 390, 385
    else:
        lag1, lag24, roll3, roll6 = 250, 240, 245, 243

    input_df = pd.DataFrame([[T_out, RH_out, Visibility, Tdewpoint,
                               hour, day_of_week, is_weekend,
                               lag1, lag24, roll3, roll6]],
                             columns=['T_out', 'RH_out', 'Visibility', 'Tdewpoint',
                                      'hour', 'day_of_week', 'is_weekend',
                                      'Appliances_lag1', 'Appliances_lag24',
                                      'Appliances_roll3', 'Appliances_roll6'])

    if st.button("⚡ Predict Energy Usage"):
        try:
            response = requests.post("https://energy-app-h5c9.onrender.com/predict",
                                     json=input_df.to_dict(orient="records")[0])
            if response.status_code == 200:
                prediction = response.json()['predicted_usage']
                st.success(f"🔌 Predicted Appliance Usage: **{prediction:.2f} Wh**")
            else:
                st.error("❌ Failed to get prediction from API.")
        except Exception as e:
            st.error(f"⚠️ API Error: {e}")

# --- Tab 2: Upload CSV Trend ---
with tabs[1]:
    st.header("📂 Analyze Past Appliance Usage")
    st.markdown("Upload a CSV file with `datetime` and `Appliances` columns to view usage trends.")

    uploaded_file = st.file_uploader("📤 Upload CSV", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=['datetime'])

        if 'datetime' in df.columns and 'Appliances' in df.columns:
            df.set_index('datetime', inplace=True)
            st.line_chart(df['Appliances'], use_container_width=True)
        else:
            st.error("CSV must contain 'datetime' and 'Appliances' columns.")

# --- Tab 3: Feature Importance ---
with tabs[2]:
    st.header("🔍 What Influences Energy Usage?")
    st.markdown("This shows which features matter most in the prediction model.")

    st.image("feature_importance.png", caption="Top 10 Features from XGBoost", use_container_width=True)

    st.markdown("""
    **📘 Feature Meaning (in simple words):**
    - `Appliances_roll3`: Avg. usage in last 3 hours
    - `Appliances_lag1`: Usage 1 hour ago
    - `hour`: Time of day (0–23)
    - `Appliances_roll6`: Avg. usage in last 6 hours
    - `Appliances_lag24`: Usage 24 hours ago
    - `RH_out`: Outdoor humidity
    - `T_out`: Outdoor temperature
    - `Tdewpoint`: Dew point temperature
    - `Visibility`: How clear the sky is (in km)
    - `day_of_week`: Day (Mon to Sun)
    """)

# --- Footer ---
st.markdown("---")
st.caption("Developed by Anangsha Halder | Smart Home AI Dashboard | © 2025")


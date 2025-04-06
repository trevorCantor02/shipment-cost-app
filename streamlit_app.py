import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
with open("shipment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Shipment Cost Predictor", page_icon="📦")
st.title("📦 Shipment Cost Predictor")

st.markdown("Enter shipment details below to estimate cost:")

# Example input fields — adjust to your dataset's features
weight = st.number_input("Weight (lbs)", min_value=1.0)
distance = st.number_input("Distance (miles)", min_value=1.0)
ship_month = st.selectbox("Month Shipped", list(range(1, 13)))
ship_weekday = st.selectbox("Day of Week Shipped (0=Mon)", list(range(7)))

# One-hot example input — expand as needed for your actual encoded fields
input_dict = {
    "Weight": weight,
    "Distance": distance,
    "ShipMonth": ship_month,
    "ShipWeekday": ship_weekday,
    # Add other one-hot fields here if needed (e.g. location dummies)
}

input_df = pd.DataFrame([input_dict])

# Apply same scaling
scaled_input = scaler.transform(input_df)

if st.button("Predict Shipment Cost"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"💰 Predicted Shipment Cost: ${prediction:,.2f}")

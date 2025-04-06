import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Streamlit UI
st.set_page_config(page_title="Shipment Cost Predictor", layout="wide")
st.title("📦 Shipment Cost Predictor")

try:
    # Load data
    df = pd.read_csv("Machine_Learning_Shipping_Dataset.csv")

    # Date processing
    df["Date Shipped"] = pd.to_datetime(df["Date Shipped"], errors='coerce')
    df["ShipMonth"] = df["Date Shipped"].dt.month
    df["ShipWeekday"] = df["Date Shipped"].dt.weekday
    df = df.drop(columns=["Date Shipped"])

    # Encode
    df_encoded = pd.get_dummies(df)

    # Features & target
    X = df_encoded.drop(columns=["Shipment Cost"])
    y = df_encoded["Shipment Cost"]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Models
    lgbm = LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, subsample=0.8, n_jobs=-1, verbose=-1)
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)

    stacked_model = StackingRegressor(
        estimators=[("lgbm", lgbm), ("rf", rf)],
        final_estimator=RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42),
        n_jobs=-1
    )

    # Train model
    with st.spinner("Training model..."):
        stacked_model.fit(X_train, y_train)
        y_pred = stacked_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

    # Show results
    st.success(f"✅ Model Trained | MSE: {mse:.2f}, R²: {r2:.4f}")

    # Plot: Prediction vs Actual
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, alpha=0.6)
    ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax1.set_xlabel("Actual Shipment Cost")
    ax1.set_ylabel("Predicted Shipment Cost")
    ax1.set_title("Prediction vs Actual")
    st.pyplot(fig1)

    # Plot: Residuals
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_pred, residuals, alpha=0.6)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Predicted Shipment Cost")
    ax2.set_ylabel("Residuals")
    ax2.set_title("Residual Plot")
    st.pyplot(fig2)

    # Cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(stacked_model, X_scaled, y, cv=kf, scoring="r2", n_jobs=-1)

    fig3, ax3 = plt.subplots()
    ax3.bar(range(1, len(cv_scores) + 1), cv_scores)
    ax3.set_ylim(0.90, 1.01)
    ax3.set_xlabel("Fold")
    ax3.set_ylabel("R² Score")
    ax3.set_title(f"Cross-validated R² Scores (Mean: {cv_scores.mean():.4f})")
    st.pyplot(fig3)

except Exception as e:
    st.error(f"❌ Something went wrong: {e}")

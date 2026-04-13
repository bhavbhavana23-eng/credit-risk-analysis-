# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Title
st.title("💳 Credit Risk Analysis System")

st.write("Enter applicant details below:")

# 🔹 INPUTS (Generic - works for German dataset)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
credit_amount = st.number_input("Credit Amount", min_value=0, value=5000)
duration = st.number_input("Loan Duration (months)", min_value=1, value=12)

# Predict button
if st.button("Predict"):

    # Create input dictionary
    input_data = {
        "Age": age,
        "Credit amount": credit_amount,
        "Duration": duration
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encoding
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns
    input_df = input_df[columns]

    # Scale data
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # Risk score
    risk_score = (1 - probability[0][1]) * 100

    # Output
    st.subheader("Result")

    if prediction[0] == 1:
        st.success("✅ Low Risk (Good Credit)")
    else:
        st.error("❌ High Risk (Bad Credit)")

    st.write(f"Risk Score: {risk_score:.2f}%")

    # Risk level
    if risk_score < 30:
        st.success("🟢 Low Risk")
    elif risk_score < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

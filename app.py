# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Title
st.title("💳 Credit Risk Analysis System")

st.write("Enter applicant details below:")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
credit_amount = st.number_input("Credit Amount", min_value=0, value=5000)
duration = st.number_input("Loan Duration (months)", min_value=1, value=12)

# Predict button
if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Credit amount': [credit_amount],
        'Duration': [duration]
    })

    # Add missing columns (VERY IMPORTANT)
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Arrange column order
    input_data = input_data[columns]

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    # Risk Score (Correct logic)
    risk_score = (1 - probability[0][1]) * 100

    # Result section
    st.subheader("Result")

    # Loan Decision (MAIN OUTPUT)
    if risk_score < 50:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    # Show Risk Score
    st.write(f"Risk Score: {risk_score:.2f}%")

    # Risk Category
    if risk_score < 30:
        st.success("🟢 Low Risk")
    elif risk_score < 70:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

   

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("💳 Credit Risk Analysis System")

st.write("Enter applicant details:")

# Inputs
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", [0,1])

if st.button("Predict"):

    input_data = pd.DataFrame({
        "ApplicantIncome":[ApplicantIncome],
        "CoapplicantIncome":[CoapplicantIncome],
        "LoanAmount":[LoanAmount],
        "Loan_Amount_Term":[Loan_Amount_Term],
        "Credit_History":[Credit_History]
    })

    # Add missing columns
    for col in columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[columns]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    risk_score = (1 - prob[0][1]) * 100

    st.subheader("Result")

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")

    st.write(f"Risk Score: {risk_score:.2f}%")

    if risk_score < 30:
        st.success("Low Risk")
    elif risk_score < 70:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")

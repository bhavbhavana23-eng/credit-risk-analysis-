# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# 1️⃣ Load dataset
df = pd.read_csv("loan_data.csv")   # or use pd.read_excel if needed

# 2️⃣ Remove Loan_ID if exists
df = df.drop("Loan_ID", axis=1, errors='ignore')

# 3️⃣ Feature Engineering (IMPORTANT FIX)
df["Loan_Income_Ratio"] = df["LoanAmount"] / (df["ApplicantIncome"] + df["CoapplicantIncome"] + 1)

# 4️⃣ Handle Missing Values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])

# 5️⃣ Encode target variable
df['Loan_Status'] = df['Loan_Status'].map({'Y':1, 'N':0})

# 6️⃣ One-hot encoding
categorical_cols = ['Gender','Married','Education','Self_Employed','Property_Area','Dependents']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 7️⃣ Split features & target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# 8️⃣ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9️⃣ Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔟 Model (IMPORTANT FIX)
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Train model
model.fit(X_train_scaled, y_train)

# 1️⃣1️⃣ Accuracy
accuracy = model.score(X_test_scaled, y_test)
print("Model Accuracy:", accuracy)

# 1️⃣2️⃣ Save files
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("✅ Model retrained and saved successfully!")

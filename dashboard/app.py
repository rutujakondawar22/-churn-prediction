import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("### Predict which customers are likely to leave — and how much revenue is at risk")

# Load model and feature names
model = pickle.load(open(r'C:\Users\Admin\churn-prediction\models\best_model.pkl', 'rb'))
feature_names = pickle.load(open(r'C:\Users\Admin\churn-prediction\models\feature_names.pkl', 'rb'))

st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 20.0, 120.0, 65.0)
total_charges = monthly_charges * tenure

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
partner = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents?", ["Yes", "No"])
paperless = st.sidebar.selectbox("Paperless Billing?", ["Yes", "No"])
senior = st.sidebar.selectbox("Senior Citizen?", ["Yes", "No"])

st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Tenure", f"{tenure} months")
col2.metric("Monthly Charges", f"${monthly_charges:.2f}")
col3.metric("Total Charges", f"${total_charges:.2f}")

# Build input with all 39 features set to 0 first
input_data = {col: 0 for col in feature_names}

# Fill in the values
input_data['gender'] = 0
input_data['SeniorCitizen'] = 1 if senior == "Yes" else 0
input_data['Partner'] = 1 if partner == "Yes" else 0
input_data['Dependents'] = 1 if dependents == "Yes" else 0
input_data['tenure'] = tenure
input_data['PhoneService'] = 1
input_data['MultipleLines'] = 0
input_data['PaperlessBilling'] = 1 if paperless == "Yes" else 0
input_data['MonthlyCharges'] = monthly_charges
input_data['TotalCharges'] = total_charges
input_data['revenue_per_tenure'] = total_charges / (tenure + 1)

input_data['InternetService_DSL'] = 1 if internet == "DSL" else 0
input_data['InternetService_Fiber optic'] = 1 if internet == "Fiber optic" else 0
input_data['InternetService_No'] = 1 if internet == "No" else 0

input_data['Contract_Month-to-month'] = 1 if contract == "Month-to-month" else 0
input_data['Contract_One year'] = 1 if contract == "One year" else 0
input_data['Contract_Two year'] = 1 if contract == "Two year" else 0

input_data['PaymentMethod_Bank transfer (automatic)'] = 1 if payment == "Bank transfer (automatic)" else 0
input_data['PaymentMethod_Credit card (automatic)'] = 1 if payment == "Credit card (automatic)" else 0
input_data['PaymentMethod_Electronic check'] = 1 if payment == "Electronic check" else 0
input_data['PaymentMethod_Mailed check'] = 1 if payment == "Mailed check" else 0

# Create dataframe in exact feature order
input_df = pd.DataFrame([input_data])[feature_names]

# Predict
prob = model.predict_proba(input_df)[0][1]
prediction = model.predict(input_df)[0]

st.markdown("---")
st.markdown("## Prediction Result")

col1, col2, col3 = st.columns(3)

if prediction == 1:
    col1.error("⚠️ HIGH CHURN RISK")
else:
    col1.success("✅ LOW CHURN RISK")

col2.metric("Churn Probability", f"{prob*100:.1f}%")
col3.metric("Monthly Revenue at Risk", f"${monthly_charges:.2f}")

st.markdown("---")
st.markdown("### 📌 Project Insights")
st.info("🔹 Model: Logistic Regression  |  ROC AUC: 0.863")
st.info("🔹 307 customers predicted to churn from test data")
st.info("🔹 Total Monthly Revenue at Risk: $24,333.95")
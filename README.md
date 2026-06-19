# Customer Churn Prediction Project

## Problem Statement
Telecom companies lose revenue when customers leave. 
This project predicts which customers are likely to churn 
and quantifies the revenue at risk.

## Results
- Model: Logistic Regression
- ROC AUC Score: 0.863
- Customers predicted to churn: 307
- Monthly Revenue at Risk: $24,333.95

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit
- Git, GitHub

## Project Structure
- notebooks/01_EDA.ipynb — Exploratory Data Analysis
- notebooks/02_cleaning.ipynb — Data Cleaning
- notebooks/03_model.ipynb — Model Building
- dashboard/app.py — Streamlit Dashboard

## Dataset
Telco Customer Churn — Kaggle (BlastChar)
7043 rows, 21 columns

## How to Run
pip install -r requirements.txt
streamlit run dashboard/app.py
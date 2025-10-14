# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Bank Churn Predictor", layout="centered")

@st.cache_resource
def load_model(path="model_pipeline.pkl"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found at {path}. Put model_pipeline.pkl in this folder.")
    model = joblib.load(path)
    return model

model = load_model()

st.title("Bank Customer Churn Predictor")
st.write("Enter customer data and get churn prediction (probability + class).")

# --- Default expected columns (adjust if your model used different names) ---
expected_cols = [
    'CreditScore','Geography','Gender','Age','Tenure','Balance',
    'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary'
]

st.markdown("**Note:** The app expects these features (names & types):")
st.write(expected_cols)

# ---------- Input form ----------
with st.form("input_form"):
    cs = st.number_input("Credit Score", min_value=0, max_value=1000, value=650)
    geo = st.selectbox("Geography", options=["France","Spain","Germany"])
    gender = st.selectbox("Gender", options=["Female","Male"])
    age = st.number_input("Age", min_value=18, max_value=120, value=30)
    tenure = st.number_input("Tenure (years)", min_value=0, max_value=50, value=3)
    balance = st.number_input("Balance", min_value=0.0, max_value=1e9, value=10000.0, step=100.0, format="%.2f")
    numprod = st.selectbox("Number of Products", options=[1,2,3,4], index=0)
    has_card = st.selectbox("Has Credit Card?", options=[0,1], index=1)
    is_active = st.selectbox("Is Active Member?", options=[0,1], index=1)
    est_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=1e9, value=50000.0, step=100.0, format="%.2f")
    submit = st.form_submit_button("Predict")

# Build input dataframe - column order & names must match training
input_df = pd.DataFrame([{
    'CreditScore': cs,
    'Geography': geo,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': numprod,
    'HasCrCard': has_card,
    'IsActiveMember': is_active,
    'EstimatedSalary': est_salary
}])

st.subheader("Input")
st.dataframe(input_df)

# ---------- Prediction ----------
def predict_df(df):
    # Verify expected columns are present
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.error(f"Input is missing columns expected by the model: {missing}")
        raise ValueError("Missing columns for prediction.")
    # Use model pipeline directly
    proba = model.predict_proba(df)[:,1][0]
    pred = int(model.predict(df)[0])
    return pred, float(proba)

if submit:
    try:
        pred, proba = predict_df(input_df)
        st.metric("Churn probability", f"{proba:.3f}")
        if pred == 1:
            st.warning("⚠️ Model predicts the customer WILL CHURN.")
        else:
            st.success("✅ Model predicts the customer will STAY.")
    except Exception as e:
        st.exception(e)

st.markdown("---")
st.write("Or upload a CSV with the same feature columns to run batch predictions.")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Uploaded shape:", batch_df.shape)
        # Check columns
        missing = [c for c in expected_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Uploaded CSV is missing columns: {missing}")
        else:
            probs = model.predict_proba(batch_df)[:,1]
            preds = model.predict(batch_df)
            out = batch_df.copy()
            out['churn_prob'] = probs
            out['churn_pred'] = preds
            st.write(out.head(20))
            st.download_button("Download predictions CSV", out.to_csv(index=False).encode('utf-8'), file_name='predictions.csv')
    except Exception as e:
        st.exception(e)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. Load the Model and Scaler
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / 'logistic_model.pkl'
scaler_path = BASE_DIR / 'scaler.pkl'

if model_path.exists() and scaler_path.exists():
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    st.error(f"Model or scaler file not found in {BASE_DIR}. Please ensure 'logistic_model.pkl' and 'scaler.pkl' exist.")
    st.stop()

# ------------------------------------------------------------------------------
# 2. App Title and Layout
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Loan Approval App", page_icon="🏦")
st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant's details below to check if the loan will be **Approved** or **Rejected**.")

# ------------------------------------------------------------------------------
# 3. Input Fields
# ------------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2)
    education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed?", ["Yes", "No"])
    income_annum = st.number_input("Annual Income", min_value=0, value=500000)
    loan_amount = st.number_input("Loan Amount Request", min_value=0, value=1000000)
    loan_term = st.number_input("Loan Term (Years)", min_value=1, max_value=30, value=10)

with col2:
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=100000)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=50000)

# ------------------------------------------------------------------------------
# 4. Processing and Prediction
# ------------------------------------------------------------------------------
if st.button("Predict Loan Status", type="primary"):
    
    # A. Encode Categorical Data (Matching your One-Hot Encoding)
    # Based on your columns: 'education_Not Graduate' and 'self_employed_Yes'
    
    # If user selects "Not Graduate", value is 1. If "Graduate", value is 0.
    education_not_grad = 1 if education == "Not Graduate" else 0
    
    # If user selects "Yes", value is 1. If "No", value is 0.
    self_employed_yes = 1 if self_employed == "Yes" else 0

    # B. Create the DataFrame with CORRECT Column Order and Names
    input_data = pd.DataFrame([[
        no_of_dependents,
        income_annum,
        loan_amount,
        loan_term,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
        education_not_grad,  # Correct encoded feature
        self_employed_yes    # Correct encoded feature
    ]], columns=[
        'no_of_dependents', 
        'income_annum', 
        'loan_amount', 
        'loan_term', 
        'cibil_score', 
        'residential_assets_value', 
        'commercial_assets_value', 
        'luxury_assets_value', 
        'bank_asset_value', 
        'education_Not Graduate', 
        'self_employed_Yes'
    ])
    
    # C. Scale the Data
    # Important: Ensure the scaler was fitted on these exact columns too!
    try:
        input_data_scaled = scaler.transform(input_data)
        
        # D. Predict
        prediction = model.predict(input_data_scaled)
        
        # E. Display Result
        st.markdown("---")
        # Check prediction output (it might be 0/1 or string depending on your model)
        if prediction[0] == 1 or str(prediction[0]).strip() == "Approved":
            st.success("✅ **Congratulations! The loan is APPROVED.**")
            st.balloons()
        else:
            st.error("❌ **Sorry, the loan is REJECTED.**")
            st.write("Try increasing the CIBIL score or reducing the loan amount.")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Debug info: Check if scaler features match input columns.")

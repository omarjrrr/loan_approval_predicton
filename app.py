import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ---------------- LOAD MODEL ----------------
model = joblib.load("logistic_pipeline.pkl")

FEATURE_COLUMNS = [
    'no_of_dependents', 'education', 'self_employed',
    'income_annum', 'loan_amount', 'loan_term',
    'cibil_score', 'residential_assets_value',
    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

# ---------------- UI ----------------
st.title("Loan Approval Predictor")

col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    income_annum = st.number_input("Annual Income", min_value=0, value=500000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=1000000)

with col2:
    loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=10)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
    residential_assets_value = st.number_input("Residential Assets Value", min_value=0, value=100000)
    commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, value=0)
    bank_asset_value = st.number_input("Bank Asset Value", min_value=0, value=50000)

# ---------------- PROFILE ----------------
profile = {
    'no_of_dependents': no_of_dependents,
    'education': education,
    'self_employed': self_employed,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    df = pd.DataFrame([profile])[FEATURE_COLUMNS]

    prob = model.predict_proba(df)[0][1]

    st.subheader("Result")

    # Progress bar
    st.progress(float(prob))

    if prob > 0.5:
        st.success(f"Loan likely to be approved: {prob*100:.1f}%")
    else:
        st.error(f"Loan likely to be rejected: {(1 - prob)*100:.1f}%")

    # ---------------- SIMPLE CHART ----------------
    st.subheader("Probability")

    chart_df = pd.DataFrame({
        "Outcome": ["Rejected", "Approved"],
        "Probability": [1 - prob, prob]
    })

    st.bar_chart(chart_df.set_index("Outcome"))

    # ---------------- FEATURE IMPORTANCE ----------------
    try:
        st.subheader("Top Features")

        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        coefs = model.named_steps['classifier'].coef_[0]

        imp = pd.Series(coefs, index=feature_names)
        imp = imp.abs().sort_values(ascending=False).head(5)

        st.bar_chart(imp)

    except:
        st.info("Feature importance not available")
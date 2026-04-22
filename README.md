# Loan Approval Prediction

## The Problem
Banks and financial institutions face a major challenge in loan approval processes — decisions are often slow, inconsistent, and influenced by human bias. This leads to two costly outcomes: approving high-risk applicants who may default, or rejecting creditworthy customers and losing business.

The goal of this project is to solve this problem by building a machine learning model that predicts loan approval decisions automatically based on applicant financial and personal data — enabling faster, fairer, and more consistent decisions.

---

## Solution
A complete end-to-end ML pipeline was built including data preprocessing, feature scaling, model training, evaluation, and deployment. Three classification models were trained and compared:

| Model               | Accuracy |  Precision   | ROC-AUC |
|---------------------|----------|--------------|---------|
| Logistic Regression | ~90%     |  92%         | ~0.90   |
| KNN                 | 91%      |  94 %        |  91
| Random Forest       | 97%      |  97%         | 97

Logistic Regression was selected as the final model due to its high interpretability, stable cross-validation performance, and strong generalization on unseen data.

---

## Business Impact
- Reduces loan default risk by identifying high-risk applicants early
- Eliminates inconsistency caused by manual review
- Speeds up the approval process significantly
- Supports data-driven decision making across credit teams

---

## Live Demo
[Try the app](https://loanapprovalpredicton-ly3ytnr8vxk2bjwwgeeyr9.streamlit.app/)

## Run Locally
```bash
git clone https://github.com/omarjrrr/loan_approval_predicton
cd loan_approval_predicton
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure
├── app.py
├── loan.ipynb
├── logistic_pipeline.pkl
├── requirements.txt
└── README.md

## Tools & Technologies
Python • Scikit-learn • Pandas • NumPy • Matplotlib • Streamlit • Joblib

---

**Author: Omar Mohammed**

If you found this project useful, feel free to star the repository and share your feedback.

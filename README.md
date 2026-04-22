🏦 Loan Approval Prediction Project
📌 Problem Statement

Manual loan approval processes in financial institutions are often time-consuming, subjective, and prone to human bias. These issues can lead to financial losses by approving high-risk applicants or missed opportunities by rejecting trustworthy customers.

This project aims to build a machine learning system that automatically predicts loan approval decisions based on applicant financial and personal data.

🎯 Project Objective

To develop and evaluate machine learning classification models that accurately predict loan approval status while maintaining a strong balance between:

Precision – minimizing risky loan approvals

Recall – avoiding rejection of eligible customers

📁 Dataset

The dataset contains applicant demographic and financial information, including:

Income

Employment status

Credit history

Asset values

Loan amount

The target variable indicates whether the loan application was approved or rejected.

🧠 Solution Overview

A complete end-to-end machine learning pipeline was implemented, including:

Data Cleaning & Preprocessing

Outlier Handling (for asset-related features)

Feature Scaling

Model Training & Evaluation

Cross-Validation to ensure generalization

Model Comparison

Final Model Saving for Deployment

🤖 Models Used

The following classification models were trained and evaluated:

Logistic Regression (baseline & interpretable model)

K-Nearest Neighbors (KNN)

Random Forest Classifier

Each model was evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC-AUC Score

Cross-Validation

📊 Final Results (Best Performing Model)

The models achieved strong and stable performance.
Logistic Regression was selected as the final model.

Accuracy: ~90%

Precision: ~93%

Recall: ~91%

F1 Score: ~92%

ROC-AUC: ~0.90

Cross-Validation Mean Score: ~0.91

✅ Key Observations

No overfitting detected

Strong generalization on unseen data

Stable performance across folds

Logistic Regression was chosen due to its:

High interpretability

Stable cross-validation performance

Better generalization compared to more complex models

💡 Business Impact

This solution helps financial institutions to:

Reduce loan default risk

Improve decision consistency

Speed up loan approval processes

Enable data-driven decision-making

🚀 Deployment & Usage

The trained model was saved using joblib and deployed using Streamlit Cloud, providing an interactive web interface for real-time loan approval predictions.

🔗 Live Demo:
https://loanapprovalpredicton-ly3ytnr8vxk2bjwwgeeyr9.streamlit.app/
├── data/
├── notebooks/
├── models/
│   └── logistic_model.pkl
├── app.py
├── requirements.txt
└── README.md


🛠️ Tools & Technologies

Python

Pandas, NumPy

Scikit-learn

Matplotlib

Streamlit

Joblib

📌 Future Improvements

Hyperparameter tuning for performance optimization

Feature importance & model explainability (SHAP, LIME)

Threshold optimization for business-specific objectives

Full production deployment using APIs and cloud services

👤 Author

Omar Mohammed

⭐ If you found this project useful, feel free to star the repository and share feedback!

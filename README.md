# 🏦 Loan Approval Prediction Project

## 📌 Problem Statement

Financial institutions face a major challenge in deciding whether to approve or reject loan applications. Manual decision-making is often:

* Time-consuming
* Subjective
* Prone to human bias

Incorrect decisions can lead to:

* **Financial loss** (approving high-risk customers)
* **Missed opportunities** (rejecting trustworthy customers)

The goal of this project is to build a **machine learning model** that automatically predicts loan approval status based on applicant financial and personal data.

---

## 🎯 Project Objective

To develop and evaluate multiple machine learning classification models that can accurately predict whether a loan should be **approved or rejected**, while maintaining a strong balance between:

* Precision (avoiding risky approvals)
* Recall (not rejecting good customers)

---

## 🧠 Solution Overview

We applied a complete **end-to-end ML pipeline**, including:

1. Data Cleaning & Preprocessing
2. Outlier Handling (for asset-related features)
3. Feature Scaling
4. Model Training & Evaluation
5. Cross-Validation to ensure generalization
6. Model Comparison
7. Final Model Saving for Deployment

---

## 🤖 Models Used

The following models were trained and evaluated:

* **Logistic Regression** (Baseline & Interpretable)
* **K-Nearest Neighbors (KNN)**
* **Random Forest Classifier**

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Cross-Validation

---

## 📊 Final Results (Best Performing Model)

The models achieved strong and stable performance:

* **Accuracy:** ~90%
* **Precision:** ~93%
* **Recall:** ~91%
* **F1 Score:** ~92%
* **ROC AUC:** ~0.90
* **Cross-Validation Mean Score:** ~0.91

These results indicate:

* No overfitting
* Good generalization
* Reliable decision-making

---

## 💡 Business Impact

This model helps financial institutions:

* Reduce loan default risk
* Improve decision consistency
* Speed up loan approval processes
* Support data-driven decision making

---

## 🚀 Deployment & Usage

The trained model is saved using `joblib` and can be easily deployed using:

* Streamlit (Interactive Web App)

---

## 🔗 Demo Link

👉 **Live Demo:** *(https://loanapprovalpredicton-ftawrcg42878hrqz6dq3za.streamlit.app/)*

---

## 📂 Project Structure

```
├── data/
├── notebooks/
├── models/
│   └── logistic_model.pkl
├── README.md
└── requirements.txt
```

---

## 🛠️ Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib
* Joblib

---

## 📌 Future Improvements

* Hyperparameter tuning
* Feature importance analysis
* Model explainability (SHAP / LIME)
* Production deployment

---

## 👤 Author

**Omar Mohammed**

---

⭐ If you found this project useful, feel free to star it and share feedback!

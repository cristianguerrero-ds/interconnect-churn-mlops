# 📊 Interconnect - Churn Prediction and Retention System

This repository contains a comprehensive Machine Learning solution for predicting customer churn at the telecommunications company Interconnect. The solution evolves from an experimental environment to a production-ready software tool through an interactive web application developed in Streamlit.

The platform allows the sales team to perform real-time diagnostics using two methods: Individual Inquiry (via interactive forms) and Bulk Upload (via batch processing of data files).

---

## 🎯 Project Impact and Objectives

The main objective is to proactively identify customers with a high propensity to abandon the company's services, enabling the deployment of timely, data-driven retention strategies.

### Expected Benefits:
* **Risk Mitigation:** Ability to retain customers before they formally cancel their service through automated alerts by segment.

* **Resource Optimization:** Drastic reduction in false operational alarms compared to traditional base models.

* **Operational Efficiency:** The bulk upload module automates the calculation of financial fees by calculating complex business rules, freeing up time for the analytics team.

---

## 🧠 Champion Model Performance (LightGBM)

After evaluating multiple classification algorithms (Logistic Regression, Decision Tree, and Random Forest), **LightGBM** was selected as the optimal model for production due to its excellent overall balance and discriminatory power:

* **AUC-ROC:** 91.1% (Exceptional ability to separate classes)
* **F1-Score:** 70.9% (Optimal balance between accuracy and sensitivity)
* **Recall:** 83.9% (Detects more than 8 out of 10 customers planning to leave)
* **Accuracy:** 61.4% (Control of false positives in production)

### 📌 Key Business Findings (Analysis by Segment)
The cross-analysis implemented within the pipeline revealed critical churn patterns:
1. **Critical Group (65.7% Churn):** New customers (0-6 months) who have subscribed to both services (Telephony and Fiber Optic Internet).

2. **High Risk Group (53.5% Churn):** New customers who subscribe exclusively to Internet.

3. **Protective Factor:** Customer tenure acts as the main stabilizer; loyal customers (24+ months) have churn rates below 21%, regardless of the complexity of their services.

---

## 📁 Project Structure

```text
interconnection-abandonment-mlops/
│
├── data/ # Original datasets (Contract, Personnel, Internet, Telephone)
├── src/ # Modular support code
│ ├── __init__.py
│ └── pipeline_config.py # Feature engineering and Scikit-Learn pipeline
│
├── .venv/ # Virtual dependency environment (Excluded in .gitignore)
├── app.py # Interactive Streamlit application (Single and Bulk)
├── train.py # Model unification, cleaning, and training script
├── requirements.txt # Libraries and Project dependencies
└── README.md#Project documentation
└── .gitignore
└── notebook # EDA and model formulation prior to production.

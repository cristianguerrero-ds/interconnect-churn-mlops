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
├── .gitignore                         # Excludes environments, local databases, and temporary caches
├── .streamlit/
│   └── config.toml                    # Theme customization (Dark & Indigo setup)
├── README.md                          # Professional project documentation
├── app.py                             # Interactive Streamlit application (Single & Bulk predictions)
├── generate_test_csv.py               # Auxiliary script to generate mock customers for bulk testing
├── modelo_interconnect.pkl            # Trained and serialized production model (Ensure it's generated!)
├── requirements.txt                   # Production dependencies and library versions
├── train.py                           # Data preparation, pipeline fitting, and serialization script
│
├── data/                              # Raw source datasets (Contract, Personal, Internet, Phone)
├── data_reentrenamiento/               # Local directory for saving newly collected feedback loops
├── notebook/                          # Development EDA and prototyping phase
└── src/                               # Modular application backend
    ├── __init__.py
    └── pipeline_config.py             # Feature engineering schemas & Scikit-Learn pipeline definitions
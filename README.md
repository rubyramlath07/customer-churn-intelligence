# Customer Churn Intelligence

An end-to-end **Customer Churn Prediction and Retention Analytics** project built with Python, Machine Learning, Explainable AI (SHAP), and Streamlit.

The project transforms customer data into churn-risk predictions and business-oriented retention insights through an interactive dashboard.

---

## 📌 Project Overview

Customer churn occurs when an existing customer stops using a company's products or services.

The goal of this project is to build a practical analytics application that can:

- Analyze historical customer churn
- Predict the probability that a customer may churn
- Segment customers by predicted risk
- Explain individual predictions using SHAP
- Estimate potential revenue exposure
- Identify customers for retention follow-up
- Provide a Customer 360 view
- Export filtered and retention-oriented reports

The application is designed as a **decision-support tool**. Predictions and revenue-at-risk values should be interpreted as model-based estimates rather than guarantees.

---

## 🎯 Problem Statement

Businesses often have large customer datasets but need a systematic way to identify customers who may be at higher risk of leaving.

This project addresses the problem by combining:

**Customer Data → Data Preparation → Feature Engineering → ML Prediction → Risk Segmentation → Explainability → Retention Insights → Reporting**

---

## 🚀 Key Features

### 1. Executive Dashboard

The dashboard provides a high-level view of the selected customer population, including:

- Total customers
- Churned customers
- High-risk customers
- Churn rate
- Estimated revenue at risk
- Executive summary
- Churn driver analysis

### 2. Customer Prediction

A user can enter customer information and receive:

- Churn probability
- Risk level
- SHAP-based explanation
- Suggested retention action
- Customer input summary

### 3. Risk Segmentation

Customers are grouped using configurable business thresholds:

| Risk Level | Churn Probability |
|---|---:|
| Low | < 40% |
| Medium | 40% – < 60% |
| High | 60% – < 80% |
| Critical | ≥ 80% |

These thresholds are **business rules configured for this application**, not universal industry standards.

### 4. Revenue at Risk

The application uses the following transparent proxy:

`Revenue at Risk = Monthly Charges × Churn Probability`

This represents an estimated revenue exposure metric. It is not a forecast of guaranteed future revenue loss.

### 5. Retention Priority Queue

Customers can be prioritized using churn probability, revenue-at-risk exposure, and customer characteristics.

Example business actions include:

- Longer-term contract discussion for month-to-month customers
- Early-tenure engagement for newer customers
- Plan/value review for customers with higher monthly charges
- Billing/payment experience review for electronic-check customers
- Proactive retention outreach for other higher-risk customers

### 6. Churn Driver Analysis

Historical churn rates are analyzed across customer dimensions such as:

- Contract
- Internet Service
- Tenure
- Monthly Charges

These are **historical associations** in the dataset and should not be interpreted as proof that a particular factor causes churn.

### 7. Customer 360

The application provides an individual customer profile containing information such as:

- Customer ID
- Contract
- Tenure
- Internet Service
- Payment Method
- Monthly Charges
- Churn Probability
- Risk Level
- Revenue at Risk
- Recommended retention action

### 8. Explainable Machine Learning

SHAP is used to explain individual predictions and show which model features contributed to the prediction.

This makes the model output easier to interpret than presenting only a probability score.

### 9. Reporting & Downloads

The application supports downloadable outputs including:

- Executive summary
- Filtered customer data
- Scored customers
- High-risk customers
- Retention priority queue
- Retention Action Center data
- Customer-level profile
- One-click reporting package

---

## 🧠 Machine Learning Workflow

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Preprocessing Pipeline
        ↓
Machine Learning Classifier
        ↓
Churn Probability
        ↓
Risk Segmentation
        ↓
SHAP Explanation
        ↓
Revenue-at-Risk Estimate
        ↓
Retention Priority
        ↓
Streamlit Dashboard
```

---

## 🔧 Feature Engineering

The project includes engineered features used by the application/model workflow, including:

- `NumAddonServices`
- `AvgMonthlySpend`
- `IsMonthToMonth`
- `IsElectronicCheck`
- `TenureBucket`

The original customer information includes service, contract, billing, tenure, and financial attributes.

---

## 📊 Main Dataset Fields

The churn data includes fields such as:

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`
- `ChurnFlag`

Additional engineered/scored fields are available in the project outputs.

---

## 🛠️ Technologies Used

### Programming & Data

- Python
- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- Trained classification pipeline
- Probability-based churn scoring

### Explainability

- SHAP

### Visualization

- Matplotlib
- Streamlit charts and dashboard components

### Application

- Streamlit
- Custom CSS

### Model & Data Storage

- Joblib
- CSV
- JSON

---

## 📁 Project Structure

```text
customer-churn-intelligence/
│
├── app.py
├── style.css
├── churn.csv
├── customers_scored.csv
├── churn_model_pipeline.joblib
├── churn_model_metadata.json
├── requirements.txt
├── index.ipynb
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application |
| `style.css` | Custom dashboard styling |
| `churn.csv` | Historical customer churn dataset |
| `customers_scored.csv` | Customer-level churn predictions |
| `churn_model_pipeline.joblib` | Saved preprocessing + ML model pipeline |
| `churn_model_metadata.json` | Model evaluation/configuration metadata |
| `requirements.txt` | Python dependencies |
| `index.ipynb` | Notebook-based development/analysis |
| `README.md` | Project documentation |

---

## 📈 Model Evaluation

The application reads model evaluation information from `churn_model_metadata.json`.

The dashboard can display:

- ROC-AUC
- PR-AUC
- Decision Threshold
- Number of Customers Scored

The values are loaded from the saved model metadata rather than being hard-coded into the dashboard.

---

## 🔍 Model Explainability with SHAP

The application creates a SHAP explainer for the trained classifier and transforms the customer data through the model's preprocessing pipeline before generating explanations.

The prediction page uses SHAP to help answer:

> **Why did the model assign this customer this level of churn risk?**

This is useful for interpreting model output and connecting predictions with customer-level business context.

---

## 💼 Business Value

The project connects machine learning with practical customer-retention workflows.

Instead of stopping at:

> "This customer has a high probability of churn."

the application continues toward:

> "Which customers should the business review, why are they being flagged, what is the estimated revenue exposure, and what retention action could be considered?"

The final output is therefore a combination of:

**Prediction + Explanation + Segmentation + Business Context + Reporting**

---

## ▶️ Installation

### 1. Clone or copy the project

Place all project files in the same folder.

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

---

## 🧪 Basic Testing

Before running the application, check the Python syntax:

```bash
python -m py_compile app.py
```

If the command returns no output, the Python file passed the syntax check.

Then start Streamlit:

```bash
streamlit run app.py
```

Test:

- Dashboard navigation
- Filters
- Customer search
- Customer prediction
- SHAP explanation
- Risk segmentation
- Revenue-at-risk calculation
- Retention Action Center
- Customer 360
- CSV downloads
- Reporting package download

---

## 📦 Requirements

The project uses the following main Python packages:

```text
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
shap
```

---

## ⚠️ Important Notes

### Risk Levels

Risk thresholds are configurable application-level business rules.

### Revenue at Risk

The metric:

```text
MonthlyCharges × ChurnProbability
```

is a simplified proxy for revenue exposure. It should not be interpreted as guaranteed lost revenue.

### Churn Drivers

Observed differences in historical churn rates show association, not causation.

### Model Predictions

A churn probability is a model output. It should be combined with business context before taking customer-retention action.

---

## 🔮 Future Enhancements

Possible future improvements include:

- Automated model retraining
- Model drift monitoring
- Data drift monitoring
- Time-based churn analysis
- Automated retention campaign integration
- Customer lifetime value modeling
- A/B testing of retention strategies
- Advanced model comparison
- Model version tracking
- Production database integration
- Cloud deployment
- Role-based access control
- Automated scheduled reporting

---

## 👩‍💻 Project Goal

This project demonstrates how a machine-learning model can be developed into a complete business analytics application rather than remaining only as a notebook experiment.

It combines:

**Python + Data Analytics + Machine Learning + Explainable AI + Streamlit + Business Intelligence**

into one end-to-end customer churn solution.

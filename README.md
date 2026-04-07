# 🏦 Loan Approval Predictor

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning system that predicts whether a loan application will be **Approved** or **Rejected** based on applicant details. Built with a Random Forest classifier optimized via GridSearchCV and served as a Flask REST API.

---

## 🎯 Key Features

- 🔄 **End-to-end ML pipeline** — Data preprocessing → Model training → API deployment
- 📊 **Model comparison** — Logistic Regression vs Random Forest
- ⚡ **Hyperparameter tuning** — GridSearchCV for optimal performance
- 🌐 **REST API** — Real-time predictions via Flask
- ✅ **Input validation** — Comprehensive field-level validation with clear error messages
- 💡 **Explainability** — Rule-based reasoning for each prediction

---

## 📊 Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression (baseline) | 78.86% |
| Random Forest (default) | 76.42% |
| **Random Forest (GridSearchCV tuned)** | **81.06%** ✅ |

**Best Parameters:** `n_estimators=200, max_depth=3, min_samples_split=5`

### Feature Importance
The top factors influencing loan approval:
1. 🏆 **Credit History** — Most important predictor
2. 💰 **Applicant Income** — Stability indicator
3. 📋 **Loan Amount** — Risk assessment factor

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| ML Libraries | scikit-learn, NumPy, Pandas |
| Web Framework | Flask |
| Production Server | Gunicorn |
| Deployment | Render |

---

## 📂 Project Structure

```
loan-approval-predictor/
├── app.py                    # Flask API server
├── retrain_model.py          # Model training pipeline
├── model.pkl                 # Trained RandomForest model
├── scaler.pkl                # StandardScaler for features
├── loan.ipynb                # Jupyter notebook (EDA & experiments)
├── loan_data.csv             # Training dataset (614 records)
├── test.http                 # API test requests
├── requirements.txt          # Python dependencies
├── Procfile                  # Deployment configuration
├── .gitignore                # Git ignore rules
└── README.md                 # Documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ramana-07-web/loan-approval-predictor.git
cd loan-approval-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the API
```bash
python app.py
```

The server starts at `http://127.0.0.1:5000/`

---

## 🌐 API Documentation

### Health Check

```
GET /
```

**Response:** `Loan Prediction API is running`

---

### Predict Loan Approval

```
POST /predict
Content-Type: application/json
```

#### Request Body

```json
{
  "Gender": 1,
  "Married": 1,
  "Dependents": 0,
  "Education": 1,
  "Self_Employed": 0,
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 0,
  "LoanAmount": 150,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": 2
}
```

#### Success Response (200)

```json
{
  "prediction": 1,
  "result": "Approved",
  "reason": "Good credit history, Stable income"
}
```

#### Error Response (400)

```json
{
  "error": "Missing field: Credit_History"
}
```

---

### Field Reference

| Field | Type | Valid Values | Description |
|-------|------|-------------|-------------|
| `Gender` | int | `0`, `1` | 0 = Female, 1 = Male |
| `Married` | int | `0`, `1` | 0 = No, 1 = Yes |
| `Dependents` | int | `0`, `1`, `2`, `3` | Number of dependents |
| `Education` | int | `0`, `1` | 0 = Not Graduate, 1 = Graduate |
| `Self_Employed` | int | `0`, `1` | 0 = No, 1 = Yes |
| `ApplicantIncome` | int | `≥ 0` | Monthly income of applicant |
| `CoapplicantIncome` | float | `≥ 0` | Monthly income of co-applicant |
| `LoanAmount` | float | `> 0` | Loan amount (in thousands) |
| `Loan_Amount_Term` | float | `> 0` | Loan term (in days) |
| `Credit_History` | int | `0`, `1` | 0 = Bad, 1 = Good |
| `Property_Area` | int | `0`, `1`, `2` | 0 = Rural, 1 = Semiurban, 2 = Urban |

---

## ⚙️ ML Pipeline

### Data Preprocessing
1. **Missing values** — Filled with median (numeric) or mode (categorical)
2. **Dependents** — Converted `"3+"` → `3` (numeric)
3. **Encoding** — LabelEncoder for categorical variables
4. **Scaling** — StandardScaler for feature normalization

### Training Pipeline
```bash
python retrain_model.py
```
This script runs the complete pipeline: preprocessing → encoding → scaling → model comparison → GridSearchCV → export best model.

---

## 🧠 Key Learnings
- Building end-to-end ML pipelines from data to deployment
- Model evaluation, comparison, and hyperparameter optimization
- Deploying ML models as production-ready REST APIs
- Input validation and error handling best practices

## 📌 Future Improvements
- [ ] Add frontend UI for user interaction
- [ ] Integrate SHAP/LIME for advanced explainability
- [ ] Improve accuracy with feature engineering
- [ ] Add Docker containerization
- [ ] Add unit tests and CI/CD pipeline

---

## 👤 Author

**Venkat Dharmapuri**

- 📧 Email: venkatdharmapuri07@gmail.com
- 🐙 GitHub: [@ramana-07-web](https://github.com/ramana-07-web)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

# 🚀 Employee Attrition Prediction with Model Stealing Analysis

An end-to-end Machine Learning project that predicts employee attrition and demonstrates a **black-box model stealing attack** against a deployed ML model.

This project combines **Machine Learning**, **Flask REST APIs**, **Web Development**, and **ML Security** concepts in a single application.

---

## 📌 Features

- Employee Attrition Prediction using Machine Learning
- Exploratory Data Analysis (EDA)
- Data Preprocessing with StandardScaler and OneHotEncoder
- Model Training using:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Hyperparameter Tuning
- Flask REST API for real-time predictions
- Interactive HR Analytics Dashboard
- Model Persistence using Joblib
- Black-box Model Stealing Attack Simulation
- Synthetic Query Generation (500+ API queries)
- Substitute Model Training
- Agreement Rate Analysis
- Security Analytics Dashboard using Chart.js

---

## 🏗️ Project Workflow

```
Employee Dataset
        │
        ▼
EDA & Preprocessing
        │
        ▼
Train ML Models
(Logistic Regression, Random Forest, XGBoost)
        │
        ▼
Best Model Selection
        │
        ▼
Flask REST API
        │
        ▼
HR Dashboard
        │
        ▼
Black-Box Attack Simulation
        │
        ▼
Stolen Dataset
        │
        ▼
Substitute Model
        │
        ▼
Agreement Analysis
```

---

## 📊 Dataset

- **Dataset Size:** 10,000 Employee Records
- **Target Variable:** Attrition (Yes / No)

### Features

- Age
- Gender
- Department
- Job Role
- Monthly Income
- Years at Company
- Overtime
- Work-Life Balance
- Job Satisfaction
- Education
- Marital Status
- Performance Rating
- and more...

---

## 🤖 Machine Learning Models

### Victim Models

- Logistic Regression
- Random Forest
- XGBoost

### Substitute Models

- Logistic Regression
- Random Forest
- XGBoost

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Agreement Rate
- Prediction Similarity

---

## 🔐 Model Stealing Attack

The project simulates a black-box attacker by:

1. Sending hundreds of synthetic employee records to the deployed API.
2. Collecting prediction responses.
3. Creating a stolen dataset.
4. Training substitute models.
5. Measuring how closely the substitute model mimics the deployed model.

---

## 🛠️ Tech Stack

### Programming

- Python

### Machine Learning

- Scikit-learn
- XGBoost
- Pandas
- NumPy

### Backend

- Flask
- REST API

### Frontend

- HTML5
- CSS3
- Bootstrap 5
- JavaScript
- Chart.js

### Visualization

- Matplotlib
- Seaborn

### Utilities

- Joblib
- Requests

---

## 📂 Project Structure

```
Employee-Attrition-Model-Stealing/
│
├── data/
├── models/
├── notebooks/
├── web/
│   ├── static/
│   ├── templates/
│   └── app.py
├── attack/
├── utils/
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

Clone the repository:

```bash
git clone https://github.com/Sthuthi1310/Employee-Attrition-Model-Stealing.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Flask application:

```bash
python web/app.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

## 📸 Screenshots

> Add screenshots of:
>
> - Home Dashboard
> - Prediction Page
> - API Response
> - Model Stealing Dashboard
> - Charts
> - Confusion Matrix

---

## 🎯 Future Improvements

- User Authentication
- Cloud Deployment
- Rate Limiting
- API Monitoring
- Model Watermarking
- Advanced ML Security Defenses

---

## 👩‍💻 Author

**Sthuthi Sheela**

GitHub: https://github.com/Sthuthi1310

---

## ⭐ If you found this project useful, consider giving it a star!
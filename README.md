# 💳 Bank Fraud Detection using Machine Learning

An AI-based fraud detection prototype that analyzes transaction information and estimates the probability of a transaction being fraudulent.

## 🌐 Live Demo

[Launch the Bank Fraud Detection App](https://01-fraud-detection.streamlit.app/)

## 📌 Project Overview

Financial transaction datasets are highly imbalanced because fraudulent transactions represent only a small portion of total transactions. This project uses machine learning to identify potentially fraudulent transactions while handling class imbalance and optimizing the classification threshold.

The project provides:

- XGBoost-based fraud classification
- Class-imbalance handling
- Decision-threshold optimization
- SHAP-based model explainability
- Fraud probability scoring
- Interactive Streamlit interface

## 🎯 Objective

Build a machine learning system that can analyze transaction characteristics and flag potentially suspicious transactions for review.

## 📊 Features Used

The model uses the following transaction features:

- Transaction amount
- Hour of the day
- Transaction type
- Merchant category
- Country risk score
- Device risk score
- IP risk score
- Account age
- Transaction velocity

## 🤖 Machine Learning Approach

The project uses **XGBoost Classifier** for fraud classification.

The training pipeline includes:

1. Synthetic transaction data generation
2. Feature preparation and encoding
3. Train/validation split
4. Class-imbalance handling using `scale_pos_weight`
5. XGBoost model training
6. Probability-based fraud scoring
7. Decision-threshold optimization
8. Model evaluation
9. SHAP explainability
10. Saving the trained model for the Streamlit application

## 📈 Model Evaluation

The model is evaluated using metrics suitable for imbalanced classification:

- ROC-AUC
- Average Precision
- Precision
- Recall
- F1-score
- False Positive Rate

The classification threshold is optimized rather than relying only on the default `0.50` probability threshold.

## 🔍 Explainability

SHAP (SHapley Additive exPlanations) is used to provide insight into which features contribute to the model's predictions.

The generated SHAP visualization is stored in:

artifacts/shap_summary.png


## 🖥️ Streamlit Application

The project includes an interactive Streamlit interface where a user can enter transaction information and receive:

Fraud probability
Risk indication
Decision threshold

Example output:

Fraud probability: 4.74%
Risk: LOW RISK
Decision threshold: 0.65

The application is intended as a demonstration of real-time transaction scoring.
 
---
## 📂 Project Structure
01_fraud_detection/
│
├── artifacts/
│   ├── fraud_model.joblib
│   └── shap_summary.png
│
├── app.py
├── train.py
├── requirements.txt
└── README.md

## ⚙️ Installation
---
Create and activate a virtual environment:

Windows
python -m venv .venv
.venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt

# 🚂 Train the Model

Run:

python train.py

The trained model will be saved inside:

artifacts/fraud_model.joblib

The SHAP explanation plot will be generated as:

artifacts/shap_summary.png

## Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in the browser and provide an interactive fraud-risk scoring interface.

## 📁 Dataset

This prototype uses synthetically generated transaction data for reproducibility and demonstration purposes.

The synthetic dataset is not a real banking dataset and should not be interpreted as representing actual financial transaction behavior.

## ⚠️ Limitations

This project is an educational/prototype implementation.

Current limitations include:

Training data is synthetic.
The model has not been validated on real-world banking transaction data.
The application is not connected to a real banking or payment system.
No production API is implemented.
No cloud deployment is included in the current prototype.
Production monitoring and automated model-drift detection are not implemented.

Therefore, the reported model performance should not be interpreted as real-world banking fraud-detection performance.

## 🔮 Future Improvements

Possible future improvements include:

Training on a real/public fraud dataset
FastAPI inference endpoint
Docker containerization
AWS EC2 deployment
AWS S3 model storage
AWS Lambda inference
Prometheus/Grafana monitoring
Model drift detection
Automated model retraining
Integration with real-time transaction streams

## 🛠️ Technology Stack
Python
Pandas
NumPy
Scikit-learn
XGBoost
SHAP
Joblib
Matplotlib
Streamlit

## ✅ Project Status

Completed — Prototype / Internship Submission

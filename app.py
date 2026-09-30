import joblib, numpy as np, pandas as pd, streamlit as st

st.set_page_config(page_title="Fraud Risk Engine", page_icon="💳", layout="centered")
st.title("💳 AI Bank Fraud Detection")
st.caption("XGBoost + class-imbalance handling + threshold optimization")

bundle = joblib.load("artifacts/fraud_model.joblib")
model, columns, threshold = bundle["model"], bundle["columns"], bundle["threshold"]

amount = st.number_input("Transaction amount", min_value=1.0, value=250.0)
hour = st.slider("Hour of day", 0, 23, 14)
transaction_type = st.selectbox("Transaction type", ["purchase","transfer","withdrawal","online"])
merchant_category = st.selectbox("Merchant category", ["retail","travel","food","electronics","gaming"])
country_risk = st.slider("Country risk", 0.0, 1.0, 0.2)
device_risk = st.slider("Device risk", 0.0, 1.0, 0.2)
ip_risk = st.slider("IP risk", 0.0, 1.0, 0.2)
account_age_days = st.number_input("Account age (days)", 1, 5000, 500)
velocity_1h = st.number_input("Transactions in last hour", 0, 50, 2)

if st.button("Analyze transaction", type="primary"):
    row = pd.DataFrame([{
        "amount": amount, "hour": hour,
        "transaction_type": transaction_type,
        "merchant_category": merchant_category,
        "country_risk": country_risk, "device_risk": device_risk,
        "ip_risk": ip_risk, "account_age_days": account_age_days,
        "velocity_1h": velocity_1h
    }])
    row = pd.get_dummies(row)
    row = row.reindex(columns=columns, fill_value=0)
    score = float(model.predict_proba(row)[0,1])
    st.metric("Fraud probability", f"{score:.2%}")
    if score >= threshold:
        st.error("⚠️ FLAGGED FOR REVIEW")
    else:
        st.success("✅ LOW RISK")
    st.info(f"Decision threshold: {threshold:.2f}")

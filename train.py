import os, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
N = 30000

df = pd.DataFrame({
    "amount": np.round(RNG.lognormal(3.2, 1.0, N), 2),
    "hour": RNG.integers(0, 24, N),
    "transaction_type": RNG.choice(["purchase","transfer","withdrawal","online"], N),
    "merchant_category": RNG.choice(["retail","travel","food","electronics","gaming"], N),
    "country_risk": RNG.random(N),
    "device_risk": RNG.random(N),
    "ip_risk": RNG.random(N),
    "account_age_days": RNG.integers(1, 2500, N),
    "velocity_1h": RNG.poisson(2, N),
})

# Synthetic fraud mechanism with <5% positives.
risk = (
    -6.0
    + 1.4*df["country_risk"]
    + 1.6*df["device_risk"]
    + 1.8*df["ip_risk"]
    + 0.00004*df["amount"]
    + 0.20*(df["velocity_1h"] >= 6)
    + 0.35*((df["hour"] <= 4) | (df["hour"] >= 23))
    + 0.35*(df["transaction_type"] == "transfer")
)
prob = 1/(1+np.exp(-risk))
df["is_fraud"] = RNG.binomial(1, prob)

# Keep a rare fraud class but ensure both classes exist.
if df["is_fraud"].mean() > 0.05:
    threshold = np.quantile(prob, 0.97)
    df["is_fraud"] = (prob >= threshold).astype(int)

X = pd.get_dummies(df.drop(columns=["is_fraud"]), drop_first=False)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scale_pos_weight = max(1, (y_train == 0).sum() / max(1, (y_train == 1).sum()))
model = XGBClassifier(
    n_estimators=250, max_depth=5, learning_rate=0.08,
    subsample=0.85, colsample_bytree=0.85,
    objective="binary:logistic", eval_metric="logloss",
    scale_pos_weight=scale_pos_weight, random_state=42
)
model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, proba)
ap = average_precision_score(y_test, proba)

# Threshold selection: maximize F1 while keeping FPR below 5% if possible.
thresholds = np.linspace(0.05, 0.95, 181)
best = (0.5, -1)
for t in thresholds:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0,1]).ravel()
    fpr = fp / max(1, tn+fp)
    precision = tp / max(1, tp+fp)
    recall = tp / max(1, tp+fn)
    f1 = 2*precision*recall/max(1e-9, precision+recall)
    if fpr <= 0.05 and f1 > best[1]:
        best = (float(t), float(f1))

threshold = best[0]
pred = (proba >= threshold).astype(int)
print("Fraud rate:", round(y.mean()*100, 3), "%")
print("ROC-AUC:", round(auc, 4))
print("Average precision:", round(ap, 4))
print("Selected threshold:", threshold)
print(classification_report(y_test, pred, digits=4))

os.makedirs("artifacts", exist_ok=True)
joblib.dump({"model": model, "columns": list(X.columns), "threshold": threshold}, "artifacts/fraud_model.joblib")

# SHAP summary plot
try:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test.sample(min(1000, len(X_test)), random_state=42))
    plt.figure()
    shap.summary_plot(sv, X_test.sample(min(1000, len(X_test)), random_state=42), show=False)
    plt.tight_layout()
    plt.savefig("artifacts/shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
except Exception as e:
    print("SHAP plot skipped:", e)

print("Saved artifacts/fraud_model.joblib")

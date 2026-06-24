"""
Run this script once to train and save the three models.
Usage: python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib
import os

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(DATA_PATH)

# ── 2. Clean ──────────────────────────────────────────────────────────────────
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)
df.drop(columns=["customerID"], inplace=True)

# Simplify multi-value No-service columns to plain No
for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
    df[col] = df[col].replace({"No phone service": "No", "No internet service": "No"})

# Binary Yes/No → 1/0
binary_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies",
               "PaperlessBilling"]
for col in binary_cols:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encode multi-class categoricals
df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=False)

# ── 3. Features / target ──────────────────────────────────────────────────────
TARGET = "Churn"
X = df.drop(columns=[TARGET])
y = df[TARGET]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# ── 4. Train/test split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 5. Preprocessor (scaler only; encoding already done above) ───────────────
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[NUMERIC_FEATURES] = scaler.fit_transform(X_train[NUMERIC_FEATURES])
X_test_scaled[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

# Save scaler + column order so the app can apply the same transform
preprocessor_data = {
    "scaler": scaler,
    "numeric_features": NUMERIC_FEATURES,
    "feature_columns": list(X.columns),
}
os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor_data, "models/preprocessor.pkl")
print("Saved preprocessor.pkl")

# ── 6. Define models ──────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
}

# ── 7. Train, evaluate, save ──────────────────────────────────────────────────
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    results[name] = {"accuracy": round(acc * 100, 2), "f1": round(f1 * 100, 2)}

    fname = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"models/{fname}")
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

# ── 8. Summary table ──────────────────────────────────────────────────────────
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(f"{'Model':<25} {'Accuracy %':>12} {'F1 Score %':>12}")
print("-"*50)
for name, scores in results.items():
    print(f"{name:<25} {scores['accuracy']:>12} {scores['f1']:>12}")

# Save results for the app
joblib.dump(results, "models/model_results.pkl")
print("\nAll models saved to models/ folder.")

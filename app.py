import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ── Load saved artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    pre  = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    lr   = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    knn  = joblib.load(os.path.join(MODELS_DIR, "knn.pkl"))
    dt   = joblib.load(os.path.join(MODELS_DIR, "decision_tree.pkl"))
    res  = joblib.load(os.path.join(MODELS_DIR, "model_results.pkl"))
    return pre, {"Logistic Regression": lr, "KNN": knn, "Decision Tree": dt}, res

try:
    preprocessor, model_map, model_results = load_artifacts()
    models_ready = True
except FileNotFoundError:
    models_ready = False

# ── Preprocessing helper ──────────────────────────────────────────────────────
def preprocess_input(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Simplify no-service labels
    for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        if col in df.columns:
            df[col] = df[col].replace({"No phone service": "No", "No internet service": "No"})

    # Binary yes/no
    binary_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines",
                   "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0}).fillna(df["gender"])

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).fillna(df["Churn"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # One-hot encode
    df = pd.get_dummies(df, columns=[c for c in ["InternetService", "Contract", "PaymentMethod"] if c in df.columns])

    # Align columns to training feature set
    feature_cols = preprocessor["feature_columns"]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    # Scale numerics
    numeric_features = preprocessor["numeric_features"]
    scaler = preprocessor["scaler"]
    df[numeric_features] = scaler.transform(df[numeric_features])

    return df


def predict(df_processed: pd.DataFrame, model_name: str):
    model = model_map[model_name]
    preds = model.predict(df_processed)
    try:
        proba = model.predict_proba(df_processed)[:, 1]
    except AttributeError:
        proba = preds.astype(float)
    return preds, proba


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Telecom Customer Churn Predictor")
st.markdown(
    "Predict whether a customer is likely to leave (churn) based on their account details. "
    "Built with **Logistic Regression**, **KNN**, and **Decision Tree** models."
)

if not models_ready:
    st.error(
        "**Models not found.** Please run `python train_models.py` inside the `hs-churn-predictor/` folder first."
    )
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict Single Customer", "📂 Batch Prediction (CSV)", "📊 Model Comparison"])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Single Prediction
# ═══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Customer Details")
    st.markdown("Fill in the form below and click **Predict** to see the result.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender          = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen  = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner         = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents      = st.selectbox("Has Dependents?", ["Yes", "No"])

    with col2:
        st.markdown("**Account Info**")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method  = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        total_charges   = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                          float(tenure * monthly_charges), step=1.0)

    with col3:
        st.markdown("**Services**")
        phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines  = st.selectbox("Multiple Lines", ["Yes", "No"])
        internet_svc    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup   = st.selectbox("Online Backup", ["Yes", "No"])
        device_prot     = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support    = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv    = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies= st.selectbox("Streaming Movies", ["Yes", "No"])

    model_choice = st.selectbox("Choose Model", list(model_map.keys()), key="single_model")

    if st.button("Predict", type="primary"):
        input_data = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": senior_citizen,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet_svc,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_prot, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }])

        processed = preprocess_input(input_data)
        pred, proba = predict(processed, model_choice)
        confidence = round(float(proba[0]) * 100, 1)

        st.divider()
        if pred[0] == 1:
            st.error(f"### ⚠️ This customer is **likely to churn**")
        else:
            st.success(f"### ✅ This customer is **not likely to churn**")

        st.metric("Churn Probability", f"{confidence}%")
        st.caption(f"Predicted using: {model_choice}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Prediction
# ═══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Upload a CSV File for Batch Predictions")
    st.markdown(
        "Upload a CSV with customer data (same format as the training dataset). "
        "A `customerID` column is optional — if present, it will be included in the output."
    )

    model_choice_batch = st.selectbox("Choose Model", list(model_map.keys()), key="batch_model")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_upload = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df_upload)} rows**. Preview:")
        st.dataframe(df_upload.head())

        if st.button("Run Predictions", type="primary"):
            ids = df_upload["customerID"] if "customerID" in df_upload.columns else None
            if "Churn" in df_upload.columns:
                df_upload = df_upload.drop(columns=["Churn"])

            try:
                processed = preprocess_input(df_upload)
                preds, proba = predict(processed, model_choice_batch)

                results_df = pd.DataFrame({
                    "Churn Prediction": ["Churn" if p == 1 else "No Churn" for p in preds],
                    "Churn Probability (%)": (proba * 100).round(1),
                })
                if ids is not None:
                    results_df.insert(0, "customerID", ids.values)

                st.success(f"Predictions complete! {int(preds.sum())} customers predicted to churn out of {len(preds)}.")
                st.dataframe(results_df)

                csv_out = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_out,
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.info("Make sure your CSV has the same column names as the training dataset.")

    st.markdown("---")
    st.caption("**Tip:** You can use the original `WA_Fn-UseC_-Telco-Customer-Churn.csv` file to test batch predictions.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ═══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("How Do the Three Models Compare?")
    st.markdown(
        "All three models were trained on the same 80% of the dataset and evaluated on the remaining 20%."
    )

    names     = list(model_results.keys())
    accuracies = [model_results[n]["accuracy"] for n in names]
    f1_scores  = [model_results[n]["f1"]       for n in names]

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy (%)", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1_scores,  width, label="F1 Score (%)",  color="#DD8452")

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison — Accuracy vs F1 Score")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.bar_label(bars1, padding=3, fmt="%.1f")
    ax.bar_label(bars2, padding=3, fmt="%.1f")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Metrics table
    st.markdown("### Detailed Scores")
    table_df = pd.DataFrame(model_results).T.reset_index()
    table_df.columns = ["Model", "Accuracy (%)", "F1 Score (%)"]
    st.dataframe(table_df, hide_index=True)

    # Plain-English model explanations
    st.markdown("---")
    st.markdown("### What is each model doing?")

    with st.expander("Logistic Regression"):
        st.markdown(
            "Logistic Regression finds a mathematical boundary between churners and non-churners. "
            "It works like a scoring system — each feature (e.g. contract type, charges) gets a weight, "
            "and the total score determines the prediction. It's simple, fast, and easy to explain."
        )
    with st.expander("K-Nearest Neighbors (KNN)"):
        st.markdown(
            "KNN looks at the **5 most similar customers** in the training data and predicts "
            "based on what the majority of those neighbours did. Think of it as: "
            "'customers who look like you mostly churned, so you probably will too.'"
        )
    with st.expander("Decision Tree"):
        st.markdown(
            "A Decision Tree asks a series of yes/no questions about the customer "
            "(e.g. 'Is the contract month-to-month? → Yes → Is tenure < 12 months? → Yes → Predict Churn'). "
            "It creates a flowchart-like structure that is very easy to visualise and understand."
        )

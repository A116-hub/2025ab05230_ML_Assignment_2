import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="ML Assignment 2",
    layout="centered"
)

st.title("ML Assignment 2 – Classification Models")
st.write(
    "Upload test data, select a trained model, "
    "and view evaluation metrics, classification report, and confusion matrix."
)

# ==============================
# DOWNLOAD TEST DATASET
# ==============================
st.subheader("📥 Download Test Dataset (For Evaluator)")

try:
    test_df = pd.read_csv("test_samples.csv")
    csv_data = test_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download test_samples.csv",
        data=csv_data,
        file_name="test_samples.csv",
        mime="text/csv"
    )
except FileNotFoundError:
    st.warning("⚠️ test_samples.csv not found in repository.")

st.divider()

# ==============================
# MODEL SELECTION
# ==============================
model_name = st.selectbox(
    "Select Classification Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

SCALER_PATH = "model/scaler.pkl"

# ==============================
# DATASET UPLOAD
# ==============================
uploaded_file = st.file_uploader(
    "Upload CSV Test Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        X = data.iloc[:, :-1]
        y_true = data.iloc[:, -1]

        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        if y_true.dtype == "object":
            y_le = LabelEncoder()
            y_true = y_le.fit_transform(y_true)

        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

        model = joblib.load(model_files[model_name])
        y_pred = model.predict(X_scaled)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr")
        else:
            auc = np.nan

        st.subheader(f"📈 Evaluation Metrics – {model_name}")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        mcc = matthews_corrcoef(y_true, y_pred)

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.3f}")
        col2.metric("AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")
        col3.metric("Precision", f"{precision:.3f}")

        col4.metric("Recall", f"{recall:.3f}")
        col5.metric("F1 Score", f"{f1:.3f}")
        col6.metric("MCC", f"{mcc:.3f}")

        st.subheader("📋 Classification Report")
        report_df = pd.DataFrame(
            classification_report(y_true, y_pred, output_dict=True)
        ).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))

        st.subheader("📊 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ An error occurred.")
        st.error(str(e))

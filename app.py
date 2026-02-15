import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Page Config ----------------
st.set_page_config(page_title="ML Assignment 2", layout="centered")

st.title("ML Assignment 2 – Classification Models")
st.write("Upload test data, select a model, and view predictions and evaluation results.")

# ---------------- Sample Data Download ----------------
st.subheader("📥 Download Sample Test Data")

sample_data = pd.DataFrame({
    "feature_1": [10, 20, 30],
    "feature_2": [1.5, 2.3, 3.1],
    "feature_3": ["A", "B", "A"],
    "target": ["Yes", "No", "Yes"]
})

csv = sample_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Sample CSV",
    data=csv,
    file_name="sample_test_data.csv",
    mime="text/csv"
)

st.divider()

# ---------------- Model Selection ----------------
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

# ---------------- Dataset Upload ----------------
uploaded_file = st.file_uploader("Upload CSV Test Dataset", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(data.head())

        # Split features and target
        X = data.iloc[:, :-1]
        y_true = data.iloc[:, -1]

        # Encode categorical FEATURES
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # Encode TARGET
        if y_true.dtype == "object":
            y_le = LabelEncoder()
            y_true = y_le.fit_transform(y_true)

        # Load scaler and scale
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

        # Load model
        model = joblib.load(model_files[model_name])

        # Predict
        y_pred = model.predict(X_scaled)

        # ---------------- Classification Report (Styled) ----------------
        st.subheader("📋 Classification Report")

        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()

        st.dataframe(
            report_df.style
            .background_gradient(cmap="Blues")
            .format("{:.2f}")
        )

        # ---------------- Confusion Matrix (Heatmap) ----------------
        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)

    except FileNotFoundError as e:
        st.error("❌ Required model or scaler file not found.")
        st.error(str(e))

    except ValueError as e:
        st.error("❌ Feature mismatch between training data and uploaded CSV.")
        st.error(str(e))

    except Exception as e:
        st.error("❌ An unexpected error occurred.")
        st.error(str(e))

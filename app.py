import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

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
    "and view predictions, confusion matrix, and classification report."
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
        # Load uploaded data
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(data.head())

        # Split features and target
        X = data.iloc[:, :-1]
        y_true = data.iloc[:, -1]

        # Encode categorical FEATURES
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # Encode TARGET if needed
        if y_true.dtype == "object":
            y_le = LabelEncoder()
            y_true = y_le.fit_transform(y_true)

        # Load scaler and scale features
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

        # Load selected model
        model = joblib.load(model_files[model_name])

        # Make predictions
        y_pred = model.predict(X_scaled)

        # ==============================
        # CLASSIFICATION REPORT
        # ==============================
        st.subheader("📋 Classification Report")

        report_dict = classification_report(
            y_true,
            y_pred,
            output_dict=True
        )

        report_df = pd.DataFrame(report_dict).transpose()

        st.dataframe(
            report_df.style
            .background_gradient(cmap="Blues")
            .format("{:.2f}")
        )

        # ==============================
        # CONFUSION MATRIX
        # ==============================
        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
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

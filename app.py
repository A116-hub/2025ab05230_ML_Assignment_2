import streamlit as st
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- Page Config ----------------
st.set_page_config(page_title="ML Assignment 2", layout="centered")

st.title("ML Assignment 2 – Classification Models")
st.write("Upload test data, select a model, and view predictions and evaluation results.")

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
        # Load dataset
        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.write(data.head())

        # Split features and target
        X = data.iloc[:, :-1]
        y_true = data.iloc[:, -1]

        # Encode categorical columns
        for col in X.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        # Load trained scaler and transform
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X)

        # Load selected model
        model = joblib.load(model_files[model_name])

        # Predict
        y_pred = model.predict(X_scaled)

        # Display results
        st.subheader("Classification Report")
        st.text(classification_report(y_true, y_pred))

        st.subheader("Confusion Matrix")
        st.write(confusion_matrix(y_true, y_pred))

    except FileNotFoundError as e:
        st.error("❌ Required model or scaler file not found.")
        st.error(str(e))

    except ValueError as e:
        st.error("❌ Feature mismatch between training data and uploaded CSV.")
        st.error(str(e))

    except Exception as e:
        st.error("❌ An unexpected error occurred.")
        st.error(str(e))

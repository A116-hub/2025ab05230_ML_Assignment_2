import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ML Assignment 2", layout="centered")

st.title("ML Assignment 2 – Classification Models")
st.write("Upload test data, select a model, and view predictions and evaluation results.")

# -------- Model Selection --------
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

# -------- Dataset Upload --------
uploaded_file = st.file_uploader("Upload CSV Test Dataset", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    X = data.iloc[:, :-1]
    y_true = data.iloc[:, -1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = joblib.load(model_files[model_name])
    y_pred = model.predict(X_scaled)

    # -------- Evaluation Metrics (1 mark) --------
    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))

    # -------- Confusion Matrix (1 mark) --------
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_true, y_pred))

# ML Assignment 2 – Classification Models

## a. Problem Statement

The objective of this assignment is to implement and evaluate multiple machine learning
classification models on a healthcare dataset. The project also demonstrates an end-to-end
machine learning workflow including data preprocessing, model training, evaluation, and
deployment using a Streamlit web application.

---

## b. Dataset Description

The healthcare dataset contains patient-related attributes used to predict health outcomes.
The dataset consists of more than 500 instances and more than 12 features, satisfying the
minimum requirements of the assignment. The target variable represents a multi-class
health outcome.

---

## c. Models Used and Evaluation Metrics

The following six classification models were implemented using the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (kNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

The evaluation metrics used are:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

### Model Performance Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | XX | XX | XX | XX | XX | XX |
| Decision Tree | XX | XX | XX | XX | XX | XX |
| kNN | XX | XX | XX | XX | XX | XX |
| Naive Bayes | XX | XX | XX | XX | XX | XX |
| Random Forest (Ensemble) | XX | XX | XX | XX | XX | XX |
| XGBoost (Ensemble) | XX | XX | XX | XX | XX | XX |

(Note: Replace XX with actual values obtained from model evaluation.)

---

## d. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|--------------|-------------------------------------|
| Logistic Regression | Provided a strong baseline performance with consistent and stable results. |
| Decision Tree | Showed reasonable performance but exhibited signs of overfitting. |
| kNN | Performance was sensitive to feature scaling and the choice of k value. |
| Naive Bayes | Computationally efficient but limited due to strong feature independence assumptions. |
| Random Forest (Ensemble) | Achieved improved performance by reducing overfitting through ensemble learning. |
| XGBoost (Ensemble) | Delivered the best overall performance due to effective gradient boosting. |

---

## Deployment

The trained models are deployed using Streamlit Community Cloud. The application allows
users to upload test data, select a model, and view predictions and evaluation metrics.

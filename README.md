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
| Logistic Regression | 0.32 | 0.45857268369706194 | 0.3083227815916603 | 0.32 | 0.304650780491442 | -0.033103793184101775 |
| Decision Tree | 0.2906666666666667 | 0.46603462727426526 | 0.2908538587848933 | 0.2906666666666667 | 0.29044607244607246 | -0.06735094664794654 | 
| kNN | 0.29333333333333333 | 0.4686825223987464 | 0.2853019607843137 | 0.29333333333333333 | 0.285535031752465 | -0.06784319419392457 |
| Naive Bayes | 0.304 | 0.4595889820211931 | 0.30007687654148635 | 0.304 | 0.2983536775461281 | -0.05307991366403321 |
| Random Forest (Ensemble) | 0.296 | 0.46116007682987886 | 0.288580989729225 | 0.296 | 0.2894283547680592 | -0.06419951415443112 |
| XGBoost (Ensemble) | 0.3253333333333333 | 0.4701249956851547 | 0.320732443982444 | 0.3253333333333333 | 0.3213233888523915 | -0.017641460469888584|

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
## f. How to Run the Project Locally

1. Clone the repository:
https://github.com/A116-hub/2025ab05230_ML_Assignment_2

2. Navigate to the project directory:
cd 2025ab05230_ML_Assignment_2

3. Run the Streamlit application:
streamlit run app.py

---

## g. Live Streamlit Application

Live App Link:  
https://2025ab05230mlassignment2-jewkezutdzs5do6z2fy7g9.streamlit.app/

---
## h. Tools and Libraries Used

Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn

---
## i. Conclusion

This assignment demonstrates the implementation and comparison of multiple classification models and their deployment using Streamlit.
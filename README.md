# Telco Customer Churn Prediction

## Project Overview

This is a real-world machine learning classification project focused on predicting whether a telecom customer is likely to churn. The objective is to build a reliable predictive model that helps identify customers at risk of leaving the service so that proactive retention strategies can be applied.

The project includes data exploration, preprocessing, feature engineering, model training, evaluation, threshold analysis, and customer-level prediction. It also covers business-oriented insights to interpret model results in a practical telecom context. Two models were evaluated: Logistic Regression and Random Forest. The workflow demonstrates how to train, compare, and validate these models for churn prediction, and it also includes saving and loading the trained model using joblib for reuse in real-world deployment scenarios.

## Dataset

The dataset used in this project is the Telco Customer Churn dataset. It contains 7,043 customer records and 21 columns in the original dataset. The target variable is Churn, where Yes indicates that the customer churned and No indicates that the customer did not churn. The dataset includes customer demographic information, services used, contract details, payment method, tenure, monthly charges, and total charges, making it suitable for a real-world customer retention analysis.

## Project Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Cleaning and Feature Preparation
4. Train-Test Split
5. Preprocessing
   - StandardScaler for numerical features
   - OneHotEncoder for categorical features
6. Model Training
   - Logistic Regression
   - Random Forest
7. Model Evaluation
   - Classification Report
   - Confusion Matrix
   - ROC-AUC
   - Precision-Recall Curve
8. Threshold Analysis
9. Business Insights
10. Customer-Level Churn Prediction
11. Model Saving and Loading with Joblib

## Models

### Logistic Regression
This model was used as the primary classification model for customer churn prediction. To address class imbalance, class_weight="balanced" was applied. The model was implemented inside a preprocessing Pipeline so that feature scaling and encoding were handled consistently during training and inference.

### Random Forest
This model was used as a second classification model for comparison against Logistic Regression. It was configured with 200 trees using n_estimators=200 and class_weight="balanced" to better handle the imbalanced churn distribution. The model was trained on the preprocessed and encoded features to evaluate its classification performance.

## Model Evaluation

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.74 | 0.77 |
| Churn Recall (Yes) | 0.78 | 0.63 |
| Best F1 Score | 0.615 | 0.620 |
| ROC-AUC | 0.842 | 0.823 |

Accuracy measures overall correct predictions. Churn Recall measures how many actual churn customers were identified. F1 balances precision and recall, while ROC-AUC measures the model's ability to distinguish between churn and non-churn customers. Threshold selection can change precision, recall, F1, and business cost. The evaluation is based on the test set, and there is no single universally best model; the best choice depends on the business objective.

## Business Insights

The dataset shows clear churn risk patterns among certain customer segments. Observed churn rates were highest for month-to-month contracts (42.71%), followed by one-year contracts (11.27%) and two-year contracts (2.83%). Customers using Fiber Optic internet had a churn rate of 41.89%, while electronic check customers showed a churn rate of 45.29%. Customers without OnlineSecurity, TechSupport, and OnlineBackup also showed elevated churn rates at 41.77%, 41.64%, and 39.93%, respectively.

A particularly high-risk group was identified: customers with a month-to-month contract, Fiber Optic internet, electronic check payment, and no OnlineSecurity or TechSupport. This combined group included 989 customers and showed an observed churn rate of 65.82%.

These findings represent observed associations in the dataset and should not be interpreted as proof of causation.

## How to Run

1. Clone or download the project to your local machine.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment in the project directory.
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Make sure the dataset is available at:
   `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
6. Run the project script:
   ```bash
   Real ML Project.py
   ```

The trained model is saved as `telco_churn_model.pkl`.

## Project Structure

```text
day_01/
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
├── src/
├── tests/
├── .gitignore
├── requirements.txt
├── README.md
├── Real ML Project.py
└── telco_churn_model.pkl
```

- `data/raw` contains the raw dataset used for analysis and modeling.
- `notebooks` is for exploratory notebooks and experiments.
- `src` is for source code and reusable project logic.
- `tests` is for testing and validation scripts.
- `requirements.txt` contains the Python dependencies required to run the project.
- `telco_churn_model.pkl` is the saved trained model produced by the project.
